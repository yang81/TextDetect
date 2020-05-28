#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"



namespace caffe {


template <typename Dtype>
__global__ void SigmoidCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, const Dtype* weight, Dtype* loss,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      loss[i] = 0;
      counts[i] = 0;
    } else {
			
			Dtype weight_ = 0;

			if (weight){
				weight_ = weight[i];
			}

			
      loss[i] = weight_ *(input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] *
          (input_data[i] >= 0))));
      counts[i] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossIgnoreDiffGPU(const int count,
    const int ignore_label, const Dtype* target, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, count) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == ignore_label) {
      diff[i] = 0;
    }
  }
}


template <typename Dtype>
__global__ void SigmoidCrossEntropyLossBalanceDiffGPU(const int count,
		const Dtype* target, Dtype* diff, const Dtype* weight) {
	CUDA_KERNEL_LOOP(i, count) {
	
		Dtype weight_ = 0;

		if (weight){
			weight_ = weight[i];
		}
		
		diff[i] *= weight_;
	}
}

template <typename Dtype>
__global__ void reset_weights(const int count, Dtype* weight,  Dtype* loss, const Dtype mean_loss) {
	CUDA_KERNEL_LOOP(i, count) {

		if (loss[i] < mean_loss){
			weight[i] = 0;
		}
	}
}



template <typename Dtype>
__global__ void OddEvenSort_kernel(const int count, Dtype* data, Dtype* index, int len, int isOdd){
	CUDA_KERNEL_LOOP(i, count) {
		Dtype d0 = data[isOdd + i*2];
		int idx0 =  static_cast<int>(index[isOdd + i*2]);
		
		if (isOdd + i*2 + 1 < len){
			Dtype d1 = data[isOdd + i*2 + 1];
			int idx1 = static_cast<int>(index[isOdd + i*2 + 1]);
		
			if (d0<d1){
				data[isOdd + i*2] = d1;
				index[isOdd + i*2] = idx1;
			
				data[isOdd + i*2 + 1] = d0;		
				index[isOdd + i*2 + 1] = idx0;
			}
		}
	}
}


template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::odd_even_sort(Blob<Dtype> *loss_index){
	const int len = loss_index->count();
	const int threads = len/2;
	
	for (int i = 0; i < len; ++i) {
		OddEvenSort_kernel<<<CAFFE_GET_BLOCKS(threads),CAFFE_CUDA_NUM_THREADS>>>(threads, 
			loss_index->mutable_gpu_data(), loss_index->mutable_gpu_diff(), len, i%2);
	}
}


template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* count_data = bottom[1]->mutable_gpu_diff();
  Dtype valid_count;


	const Dtype *weight_data = NULL;
	Dtype* m_weight_data = NULL;
	if (bottom.size()==3){
		weight_data = bottom[2]->gpu_data();
		m_weight_data = bottom[2]->mutable_gpu_data();
	}

	
	
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, weight_data, loss_data,
      has_ignore_label_, ignore_label_, count_data);
  // Only launch another CUDA kernel if we actually need the valid count.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(count, count_data, &valid_count);
  } else {
    valid_count = count;
  }
  Dtype loss;
  caffe_gpu_asum(count, loss_data, &loss);
  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;

	bool use_OHEM = false;
	if (use_OHEM){
		Blob<Dtype> loss_index;
		loss_index.ReshapeLike(*bottom[0]);

		caffe_copy(count, bottom[0]->gpu_diff(), loss_index.mutable_gpu_data());
		
		Dtype *cpu_index = loss_index.mutable_cpu_diff();
		for (int i = 0; i < count; ++i) {
			cpu_index[i] = i; 
		}
		loss_index.gpu_diff();

		odd_even_sort(&loss_index);

		Dtype * weight_data_cpu = bottom[2]->mutable_cpu_data();
			
		for (int i = 0; i < count/128; ++i) {
			int idx = static_cast<int>(cpu_index[i]);
			weight_data_cpu[idx] = 1;
		}

		//copy cpu data to gpu memory
		bottom[2]->gpu_data();
	}

	bool use_meanOHEM = false;
	if (use_meanOHEM){
		Dtype mean_loss = loss/count;
		reset_weights<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, m_weight_data, loss_data, mean_loss);
		
	}
	/*
	Frcnn::StructData<Dtype> * im_info_boxes = NULL;
	Frcnn::CopyBlobToPointer(*bottom[3], im_info_boxes);
	if (loss/normalizer_ > 0.00999999){
		printf("image name: %s",im_info_boxes->image_name_.c_str());
	}*/
	
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Zero out gradient of ignored targets.
    if (has_ignore_label_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      SigmoidCrossEntropyLossIgnoreDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, ignore_label_, target, bottom_diff);
    }

		const Dtype *weight_data = NULL;
		if (bottom.size()==3){
			weight_data = bottom[2]->gpu_data();
		}

		SigmoidCrossEntropyLossBalanceDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, target, bottom_diff, weight_data);
		
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_gpu_scal(count, loss_weight, bottom_diff);

		//printf("%p  ", bottom[0]);
		/*
		for (int i = 0; i < count; ++i) {
			printf("%f  ", bottom[0]->cpu_diff()[i]);
		}*/
		//estamate
		/*
		Dtype sum_d = 0;
		caffe_gpu_asum(count, bottom_diff, &sum_d);
		printf("crossentropy sum : %f", sum_d/normalizer_);*/
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
