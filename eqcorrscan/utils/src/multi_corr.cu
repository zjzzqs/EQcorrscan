/*
 * =====================================================================================
 *
 *       Filename:  multi_corr.cu
 *
 *        Purpose:  Routines for computing cross-correlations on GPUs
 *
 *        Created:  03/07/17 02:25:07
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  YOUR NAME (Calum Chamberlain),
 *   Organization:  EQcorrscan
 *      Copyright:  EQcorrscan developers.
 *        License:  GNU Lesser General Public License, Version 3
 *                  (https://www.gnu.org/copyleft/lesser.html)
 *
 * =====================================================================================
 */

#include <libutils_gpu.h>
#include <cuda.h>
#include <cufft.h>
#include <cufftw.h>


static inline void set_ncc(
    long t, long i, int chan, int n_chans, long template_len, long image_len,
    float value, int *used_chans, int *pad_array, float *ncc, int stack_option,
    int status);

// TODO: This should be a GPU function
int normxcorr_cuda_internal(
    long template_len, long n_templates, float *image, long image_len,
    int chan, int n_chans, float *ncc, long ncc_len, long fft_len,
    float *template_ext, float *image_ext, float *norm_sums, float *ccc,
    fftwf_complex *outa, fftwf_complex *outb, fftwf_complex *out,
    fftwf_plan pb, fftwf_plan px, int *used_chans, int *pad_array,
    int num_threads, int *variance_warning, int *missed_corr,
    int stack_option, long offset, int status)
{
  /*
    Internal function for chunking cross-correlations
    template_len:   Length of template
    n_templates:    Number of templates
    image:          Image signal (to scan through) - in this case this is a pointer to the starting index of the
                    image for this chunk
    image_len:      Length of image chunk (not complete length of image)
    chan:           Channel number - used for stacking, otherwise set to 0
    n_chans:        Number of channels - used for stacking, otherwise set to 1
    ncc:            Output for cross-correlation - should be pointer to memory. This should be the whole ncc, not just
                    the ncc starting at this chunk because padding requires negative indexing.
                    Shapes and output determined by stack_option:
        1:          Output stack correlograms, ncc must be
                    (n_templates x image_len - template_len + 1) long.
        0:          Output individual channel correlograms, ncc must be
                    (n_templates x image_len - template_len + 1) long and initialised
                    to zero before passing into this function.
    ncc_len:        Total length of the ncc (not just the chunk).
    fft_len:        Size for fft
    template_ext:   Input FFTW array for template transform (must be allocated)
    image_ext:      Input FFTW array for image transform (must be allocated)
    norm_sums:      Normalised, summed templates
    ccc:            Output FFTW array for reverse transform (must be allocated)
    outa:           Output FFTW array for template transform (must be computed)
    outb:           Output FFTW array for image transform (must be allocated)
    out:            Input array for reverse transform (must be allocated)
    pb:             Forward plan for image
    px:             Reverse plan
    used_chans:     Array to fill with number of channels used per template - must
                    be n_templates long
    pad_array:      Array of pads, should be n_templates long
    num_threads:    Number of threads to parallel internal calculations over
    variance_warning: Pointer to array to store warnings for variance issues
    missed_corr:    Pointer to array to store warnings for unused correlations
    stack_option:   Whether to stacked correlograms (1) or leave as individual channels (0),
    offset:         Offset for position of chunk in ncc (for a pad of zero).
  */
    long i, t, startind;
    long N2 = fft_len / 2 + 1;
    int unused_corr = 0;
    int * flatline_count = (int *) calloc(image_len - template_len + 1, sizeof(int));
    double *mean, *var;
    double new_samp, old_samp, sum=0.0;

    // Compute fft of image
    // TODO: This should be a cudaFFT function
    fftwf_execute_dft_r2c(pb, image_ext, outb);

    //  Compute dot product
    // TODO: Use a cuda kernal
    #pragma omp parallel for num_threads(num_threads) private(i)
    for (t = 0; t < n_templates; ++t){
        for (i = 0; i < N2; ++i)
        {
            out[(t * N2) + i][0] = outa[(t * N2) + i][0] * outb[i][0] - outa[(t * N2) + i][1] * outb[i][1];
            out[(t * N2) + i][1] = outa[(t * N2) + i][0] * outb[i][1] + outa[(t * N2) + i][1] * outb[i][0];
        }
    }

    //  Compute inverse fft
    // TODO: This should be a cudaFFT function
    fftwf_execute_dft_c2r(px, out, ccc);
 
    // Allocate mean and var arrays
    // TODO: keep this on CPU - this can be done before synchronising threads for above correlation.
    mean = (double*) malloc((image_len - template_len + 1) * sizeof(double));
    if (mean == NULL) {
        printf("Error allocating mean in normxcorr_fftw_internal\n");
        free(norm_sums);
        return 1;
    }
    var = (double*) malloc((image_len - template_len + 1) * sizeof(double));
    if (var == NULL) {
        printf("Error allocating var in normxcorr_fftw_internal\n");
        free(norm_sums);
        free(mean);
        return 1;
    }
    
    // Procedures for normalisation
    // Compute starting mean, will update this
    sum = 0.0;
    for (i=0; i < template_len; ++i){
        sum += (double) image[i];
    }
    mean[0] = sum / template_len;

    // Compute starting standard deviation
    sum = 0.0;
    for (i=0; i < template_len; ++i){
        sum += pow((double) image[i] - mean[0], 2) / (template_len);
    }
    var[0] = sum;

    // Used for centering - taking only the valid part of the cross-correlation
    startind = template_len - 1;
    if (var[0] >= ACCEPTED_DIFF) {
        double stdev = sqrt(var[0]);
        for (t = 0; t < n_templates; ++t){
            double c = ((ccc[(t * fft_len) + startind] / (fft_len * n_templates)) - norm_sums[t] * mean[0]);
            c /= stdev;
            status += set_ncc(t, offset, chan, n_chans, template_len, ncc_len,
                              (float) c, used_chans, pad_array, ncc, stack_option);
        }
        if (var[0] <= WARN_DIFF){
            variance_warning[0] = 1;
        }
    } else {
        unused_corr += 1;
    }

    // pre-compute the mean and var so we can parallelise the calculation
    for(i = 1; i < (image_len - template_len + 1); ++i){
        // Need to cast to double otherwise we end up with annoying floating
        // point errors when the variance is massive - collecting fp errors.
        new_samp = (double) image[i + template_len - 1];
        old_samp = (double) image[i - 1];
        mean[i] = mean[i - 1] + (new_samp - old_samp) / template_len;
        var[i] = var[i - 1] + (new_samp - old_samp) * (new_samp - mean[i] + old_samp - mean[i - 1]) / (template_len);
        if (new_samp == (double) image[i + template_len - 2]) {
            flatline_count[i] = flatline_count[i - 1] + 1;
        }
        else {
            flatline_count[i] = 0;
        }
    }

    // Center and divide by length to generate scaled convolution
    // TODO: This should be a CUDA kernel
    #pragma omp parallel for reduction(+:status,unused_corr) num_threads(num_threads) private(t)
    for(i = 1; i < (image_len - template_len + 1); ++i){
        if (var[i] >= ACCEPTED_DIFF && flatline_count[i] < template_len - 1) {
            double stdev = sqrt(var[i]);
            double meanstd = fabs(mean[i] * stdev);
            if (meanstd >= ACCEPTED_DIFF){
                for (t = 0; t < n_templates; ++t){
                    double c = ((ccc[(t * fft_len) + i + startind] / (fft_len * n_templates)) - norm_sums[t] * mean[i]);
                    c /= stdev;
                    status += set_ncc_gpu(t, i + offset, chan, n_chans, template_len,
                                          ncc_len, (float) c, used_chans,
                                          pad_array, ncc, stack_option);
                }
            }
            else {
                unused_corr += 1;
            }
            if (var[i] <= WARN_DIFF){
                variance_warning[0] += 1;
            }
        } else {
            unused_corr += 1;
        }
    }
    missed_corr[0] += unused_corr;

    //  Clean up
    free(mean);
    free(var);
    free(flatline_count);
}


int normxcorr_cuda_main(
    float *templates, long template_len, long n_templates, float *image,
    long image_len, int chan, int n_chans, float *ncc, long fft_len,
    float *template_ext, float *image_ext, float *ccc, fftwf_complex *outa,
    fftwf_complex *outb, fftwf_complex *out, fftwf_plan pa, fftwf_plan pb,
    fftwf_plan px, int *used_chans, int *pad_array, int num_threads,
    int *variance_warning, int *missed_corr, int stack_option) {
  /*
  Purpose: compute frequency domain normalised cross-correlation of real data using fftw
  for a single-channel
  Author: Calum J. Chamberlain
  Date: 12/06/2017
  Args:
    templates:      Template signals
    template_len:   Length of template
    n_templates:    Number of templates (n0)
    image:          Image signal (to scan through)
    image_len:      Length of image
    ncc:            Output for cross-correlation - should be pointer to memory.
                    Shapes and output determined by stack_option:
        1:          Output stack correlograms, ncc must be
                    (n_templates x image_len - template_len + 1) long.
        0:          Output individual channel correlograms, ncc must be
                    (n_templates x image_len - template_len + 1) long and initialised
                    to zero before passing into this function.
    fft_len:        Size for fft (n1)
    template_ext:   Input FFTW array for template transform (must be allocated)
    image_ext:      Input FFTW array for image transform (must be allocated)
    ccc:            Output FFTW array for reverse transform (must be allocated)
    outa:           Output FFTW array for template transform (must be allocatd)
    outb:           Output FFTW array for image transform (must be allocated)
    out:            Input array for reverse transform (must be allocated)
    pa:             Forward plan for templates
    pb:             Forward plan for image
    px:             Reverse plan
    used_chans:     Array to fill with number of channels used per template - must
                    be n_templates long
    pad_array:      Array of pads, should be n_templates long
    num_threads:    Number of threads to parallel internal calculations over
    variance_warning: Pointer to array to store warnings for variance issues
    missed_corr:    Pointer to array to store warnings for unused correlations
    stack_option:   Whether to stacked correlograms (1) or leave as individual channels (0),
  */
    long i, t, chunk, n_chunks, chunk_len, startind, step_len;
    int status = 0;
    float * norm_sums = (float *) calloc(n_templates, sizeof(float));

    if (norm_sums == NULL) {
        printf("Error allocating norm_sums in normxcorr_fftw_main\n");
        return 1;
    }

    // zero padding - and flip template
    for (t = 0; t < n_templates; ++t){
        for (i = 0; i < template_len; ++i)
        {
            template_ext[(t * fft_len) + i] = templates[((t + 1) * template_len) - (i + 1)];
            norm_sums[t] += templates[(t * template_len) + i];
        }
    }

    //  Compute fft of template
    fftwf_execute_dft_r2c(pa, template_ext, outa);

    if (fft_len >= image_len){
        n_chunks = 1;
        chunk_len = image_len;
        step_len = chunk_len;
    } else {
        chunk_len = fft_len;
        step_len = fft_len - (template_len - 1);
        n_chunks = (image_len - chunk_len) / step_len + ((image_len - chunk_len) % step_len > 0);
        if (n_chunks * step_len < image_len){n_chunks += 1;}
    }
    for (chunk = 0; chunk < n_chunks; ++chunk){
        startind = chunk * step_len;
        if (startind + chunk_len > image_len){
            chunk_len = image_len - startind;}

        memset(image_ext, 0, (size_t) fft_len * sizeof(float));
        for (i = 0; i < chunk_len; ++i){image_ext[i] = image[startind + i];}
        status += normxcorr_fftw_internal(
            template_len, n_templates, &image[startind], chunk_len, chan,
            n_chans, &ncc[0], image_len, fft_len, template_ext,
            image_ext, norm_sums, ccc, outa, outb, out, pb, px, used_chans,
            pad_array, num_threads, variance_warning, missed_corr,
            stack_option, startind);
    }
    free(norm_sums);
    return status;
}


static inline void set_ncc_gpu(
    long t, long i, int chan, int n_chans, long template_len, long image_len,
    float value, int *used_chans, int *pad_array, float *ncc, int stack_option){

    int status = 0;

    if (used_chans[t] && (i >= pad_array[t])) {
        size_t ncc_index = (t * n_chans * ((size_t) image_len - template_len + 1)) +
            (chan * ((size_t) image_len - template_len + 1) + i - pad_array[t]);

        if (isnanf(value)) {
            // set NaNs to zero
            value = 0.0;
        }
        else if (fabsf(value) > 1.01) {
            // this will raise an exception when we return to Python
//            printf("Correlation out of range at:\n\tncc_index: %ld\n\ttemplate: %ld\n\tindex: %ld\n\tvalue: %f\n",
//                   ncc_index, t, i, value);
            status = 1;
        }
        else if (value > 1.0) {
            value = 1.0;
        }
        else if (value < -1.0) {
            value = -1.0;
        }
        if (stack_option == 1){
            #pragma omp atomic  // TODO: This needs to be an atomicAdd() from Cuda
            ncc[ncc_index] += value;
        } else if (stack_option == 0){
            ncc[ncc_index] = value;
        } else {status = 2;}
    }
    return status;
}


int multi_normxcorr_cuda(float *templates, long n_templates, long template_len, long n_channels,
                         float *image, long image_len, float *ncc, long fft_len, int *used_chans,
                         int *pad_array, int num_threads_inner, int *variance_warning, int *missed_corr,
                         int stack_option)
    {
    int i, chan, n_chans, num_threads_outer=1;
    int r = 0;
    size_t N2 = (size_t) fft_len / 2 + 1;
    float **template_ext = NULL;
    float **image_ext = NULL;
    float **ccc = NULL;
    int * results = (int *) calloc(n_channels, sizeof(int));
    cufftComplex **outa = NULL;
    cufftComplex **outb = NULL;
    cufftComplex **out = NULL;
    cufftHandle pa, pb, px;

    /* num_threads_outer cannot be greater than the number of channels */
    num_threads_outer = (num_threads_outer > n_channels) ? n_channels : num_threads_outer;

    /* Outer loop parallelism seems to cause issues on OSX */
    if (OUTER_SAFE != 1 && num_threads_outer > 1){
        printf("WARNING\tMULTI_NORMXCORR_CUDA\tOuter loop threading disabled for this system\n");
        num_threads_inner *= num_threads_outer;
        printf("WARNING\tMULTI_NORMXCORR_CUDA\tSetting inner threading to %i and outer threading to 1\n", num_threads_inner);
        num_threads_outer = 1;
    }
    if (num_threads_outer > 1) {
        /* explicitly enable nested OpenMP loops */
        omp_set_nested(1);
    }

    /* allocate memory for all threads here */
    template_ext = (float**) cudaMallocHost(num_threads_outer * sizeof(float*));
    image_ext = (float**) cudaMallocHost(num_threads_outer * sizeof(float*));
    ccc = (float**) cudaMallocHost(num_threads_outer * sizeof(float*));
    outa = (cufftComplex**) cudaMallocHost(num_threads_outer * sizeof(cufftComplex*));
    outb = (cufftComplex**) cudaMallocHost(num_threads_outer * sizeof(cufftComplex*));
    out = (cufftComplex**) cudaMallocHost(num_threads_outer * sizeof(cufftComplex*));

    // All memory allocated with `fftw_malloc` to ensure 16-byte aligned.
    for (i = 0; i < num_threads_outer; i++) {
        /* initialise all to NULL so that freeing on error works */
        template_ext[i] = NULL;
        image_ext[i] = NULL;
        ccc[i] = NULL;
        outa[i] = NULL;
        outb[i] = NULL;
        out[i] = NULL;

        /* allocate template_ext arrays */
        template_ext[i] = (float*) fftwf_malloc((size_t) fft_len * n_templates * sizeof(float));
        /* allocate image_ext arrays */
        image_ext[i] = (float*) fftwf_malloc(fft_len * sizeof(float));
        /* allocate ccc arrays */
        ccc[i] = (float*) fftwf_malloc((size_t) fft_len * n_templates * sizeof(float));
        /* allocate outa arrays */
        outa[i] = (fftwf_complex*) fftwf_malloc((size_t) N2 * n_templates * sizeof(fftwf_complex));
        /* allocate outb arrays */
        outb[i] = (fftwf_complex*) fftwf_malloc((size_t) N2 * sizeof(fftwf_complex));
        /* allocate out arrays */
        out[i] = (fftwf_complex*) fftwf_malloc((size_t) N2 * n_templates * sizeof(fftwf_complex));
    }

    // We create the plans here since they are not thread safe.
    cufftPlan2d(&pa, n_templates, fft_len, template_ext[0], outa[0], FFTW_ESTIMATE);
    cufftPlan1d(&pb, fft_len, image_ext[0], outb[0], FFTW_ESTIMATE);
    cufftPlan2d(&px, n_templates, fft_len, out[0], ccc[0], FFTW_ESTIMATE);

    /* loop over the channels: note that outer threading is disabled */
    /* #pragma omp parallel for num_threads(num_threads_outer) */
    for (i = 0; i < n_channels; ++i){
        int tid = 0; /* each thread has its own workspace */

        #ifdef N_THREADS
        /* get the id of this thread */
        tid = omp_get_thread_num();
        #endif
        /* initialise memory to zero */

        if (stack_option == 1){
            chan = 0;
            n_chans = 1;
        } else {
            chan = i;
            n_chans = n_channels;
        }
        /* call the routine */
        //TODO: Should this be a cuda command?
        memset(template_ext[tid], 0, (size_t) fft_len * n_templates * sizeof(float));
        // Done internally now. memset(image_ext[tid], 0, (size_t) fft_len * sizeof(float));
        results[i] = normxcorr_cuda_main(&templates[(size_t) n_templates * template_len * i], template_len,
                                 n_templates, &image[(size_t) image_len * i], image_len, chan, n_chans, ncc, fft_len,
                                 template_ext[tid], image_ext[tid], ccc[tid], outa[tid], outb[tid], out[tid],
                                 pa, pb, px, &used_chans[(size_t) i * n_templates],
                                 &pad_array[(size_t) i * n_templates], num_threads_inner, &variance_warning[i],
                                 &missed_corr[i], stack_option);
        if (results[i] != 0){
            printf("Some error on channel %i, status: %i\n", i, results[i]);
        }
    }

    // Conduct error handling
    for (i = 0; i < n_channels; ++i){
        r += results[i];
    }
    free(results);
    /* free memory */
    cufftDestroy(px);
    cufftDestroy(pa);
    cufftDestroy(pb);

    return r;
}