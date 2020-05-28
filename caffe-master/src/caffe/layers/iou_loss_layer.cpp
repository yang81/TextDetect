/******************************************************

	File Name       : iou_loss_layer.cpp
	File Description: CPU implement
	Author          : yanghongyu
	Create Time     : 2019-9-7 9:58:26

*******************************************************/
#include <algorithm>
#include <vector>
	
#include "caffe/layers/iou_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
	
namespace caffe {
	
	template <typename Dtype>
	void IOULossLayer<Dtype>::LayerSetUp(
			const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);

	}
	
	template <typename Dtype>
	void IOULossLayer<Dtype>::Reshape(
			const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);

		CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
				"IOU_LOSS layer inputs must have the same count.";
	}
	
	template <typename Dtype>
	void IOULossLayer<Dtype>::Forward_cpu(
			const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		NOT_IMPLEMENTED;
	}
	
	template <typename Dtype>
	void IOULossLayer<Dtype>::Backward_cpu(
			const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;

		}

	
#ifdef CPU_ONLY
	STUB_GPU(IOULossLayer);
#endif
	
	INSTANTIATE_CLASS(IOULossLayer);
	REGISTER_LAYER_CLASS(IOULoss);
	
	}  // namespace caffe

