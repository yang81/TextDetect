#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/unpooling_layer.hpp"
// #include "caffe/layers/conv_layer.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/util/math_functions.hpp"
// #include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxUnpoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int unpooled_height, const int unpooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, Dtype* top_data,
    const Dtype* bottom_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int uph = max(0, min(ph * stride_h - pad_h, unpooled_height - 1));
    int upw = max(0, min(pw * stride_w - pad_w, unpooled_width - 1));
    int unpooled_index = uph * unpooled_width + upw;

    top_data += (n * channels + c) * unpooled_height * unpooled_width;
    if (bottom_mask) {
      const int mask_index = bottom_mask[index];
      top_data[mask_index] = bottom_data[index];
    } else {
      top_data[unpooled_index] = bottom_data[index];
    }
  }
}

template <typename Dtype>
__global__ void AveUnpoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int unpooled_height,
    const int unpooled_width, const int height, const int width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, unpooled_height + pad_h);
    int wend = min(wstart + kernel_w, unpooled_width + pad_w);
    int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, unpooled_height);
    wend = min(wend, unpooled_width);
		
    top_data += (n * channels + c) * unpooled_height * unpooled_width;
		//bottom_data += (n * channels + c) * height * width;
		
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
				top_data[h * unpooled_width + w] += bottom_data[index]/pool_size;
      }
    }
		
  }
}

template <typename Dtype>
__global__ void TileUnpoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int unpooled_height,
    const int unpooled_width, const int height, const int width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
		int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, unpooled_height + pad_h);
    int wend = min(wstart + kernel_w, unpooled_width + pad_w);
    //int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, unpooled_height);
    wend = min(wend, unpooled_width);
		
    top_data += (n * channels + c) * unpooled_height * unpooled_width;
		//bottom_data += (n * channels + c) * height * width;
		
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
				top_data[h * unpooled_width + w] += bottom_data[index];
      }
    }
  }
}

template <typename Dtype>
void UnpoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  int count = bottom[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0.), top_data);
	
  // We'll get the mask from bottom[2] if it's of size >2.
  //const bool use_bottom_mask = bottom.size() > 2;
	const bool use_bottom_mask = bottom.size() > 2;
	
  const Dtype* bottom_mask = NULL;
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnpoolingParameter_UnpoolMethod_MAX:
    if (use_bottom_mask) {
      bottom_mask = bottom[2]->gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxUnpoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, unpooled_height_, unpooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        bottom_mask);
    break;
  case UnpoolingParameter_UnpoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators
    AveUnpoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        unpooled_height_, unpooled_width_, height_, width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    break;
  case UnpoolingParameter_UnpoolMethod_TILE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    TileUnpoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        unpooled_height_, unpooled_width_, height_, width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxUnpoolBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_mask, const int num, const int channels,
    const int height, const int width, const int unpooled_height,
    const int unpooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int uph = max(0, min(ph * stride_h - pad_h, unpooled_height - 1));
    int upw = max(0, min(pw * stride_w - pad_w, unpooled_width - 1));
    int unpooled_index = uph * unpooled_width + upw;

    top_diff += (n * channels + c) * unpooled_height * unpooled_width;
    if (bottom_mask) {
      const int mask_index = bottom_mask[index];
      bottom_diff[index] = top_diff[mask_index];
    } else {
      bottom_diff[index] = top_diff[unpooled_index];
    }
  }
}


template <typename Dtype>
__global__ void AveUnpoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int unpooled_height,
    const int unpooled_width, const int height, const int width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, unpooled_height + pad_h);
    int wend = min(wstart + kernel_w, unpooled_width + pad_w);
    int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, unpooled_height);
    wend = min(wend, unpooled_width);
    Dtype gradient = 0;
    top_diff += (n * channels + c) * unpooled_height * unpooled_width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        gradient += top_diff[h * unpooled_width + w];
      }
    }
    bottom_diff[index] = gradient / pool_size;
  }
}

template <typename Dtype>
__global__ void TileUnpoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int unpooled_height,
    const int unpooled_width, const int height, const int width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, unpooled_height + pad_h);
    int wend = min(wstart + kernel_w, unpooled_width + pad_w);
    int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, unpooled_height);
    wend = min(wend, unpooled_width);
    Dtype gradient = 0;
    top_diff += (n * channels + c) * unpooled_height * unpooled_width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        gradient += top_diff[h * unpooled_width + w];
      }
    }
    bottom_diff[index] = gradient / pool_size;
  }
}


template <typename Dtype>
void UnpoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
	
  // We'll get the mask from bottom[1] if it's of size >2.
  //const bool use_bottom_mask = bottom.size() > 2;
	const bool use_bottom_mask = bottom.size() > 2;
	
  const Dtype* bottom_mask = NULL;
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnpoolingParameter_UnpoolMethod_MAX:
    if (use_bottom_mask) {
      bottom_mask = bottom[2]->gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxUnpoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_mask, top[0]->num(), channels_,
        height_, width_, unpooled_height_, unpooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    break;
  case UnpoolingParameter_UnpoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AveUnpoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), top_diff, top[0]->num(), channels_,
        unpooled_height_, unpooled_width_, height_, width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    break;
  case UnpoolingParameter_UnpoolMethod_TILE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    TileUnpoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), top_diff, top[0]->num(), channels_,
        unpooled_height_, unpooled_width_, height_, width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(UnpoolingLayer);


}  // namespace caffe
