/******************************************************

	File Name       : iou_loss_layer.hpp
	File Description: Head file
	Author          : yanghongyu
	Create Time     : 2019-9-7 9:57:06

*******************************************************/

#ifndef CAFFE_IOU_LOSS_LAYER_HPP_
#define CAFFE_IOU_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


template <typename Dtype>
class IOULossLayer : public LossLayer<Dtype> {
 public:
  explicit IOULossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {
			}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	
	virtual inline int ExactNumBottomBlobs() const { return 3; }

  virtual inline const char* type() const { return "IOULoss"; }

 protected:
 
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Dtype valid_count_;


  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;

};

}  // namespace caffe

#endif

