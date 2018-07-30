#include <cstdio>

#include "boost/algorithm/string.hpp"


#include "caffe/caffe.hpp"
#include "ristretto/quantization.hpp"
/////  支持 检测输出 的头文件 使用map存储结果 进行分析
#include <map>
#include "caffe/util/bbox_util.hpp"

using caffe::Caffe;
using caffe::Net;
using caffe::string;
using caffe::vector;
using caffe::Blob;
using caffe::LayerParameter;
using caffe::NetParameter;
//////////////////////
using namespace std;

// 对象  构造函数
Quantization::Quantization(string model, string weights, string model_quantized,
      int iterations, string trimming_mode, double error_margin, string gpus, string net_type) {
  this->model_   = model;  // 原模型 prototxt 框架文件
  this->weights_ = weights;// 对应权重
  this->model_quantized_ = model_quantized;// 量化后的  prototxt 框架文件 加入了量化卷积层等
  this->iterations_ = iterations;// 量化迭代次数
  this->trimming_mode_ = trimming_mode;// 量化方法
  this->error_margin_ = error_margin;// 误差赋值裕度 %
  this->gpus_ = gpus;// 使用的 gpu
  
  /////////////////////////////////
  this->net_type_ = net_type;// 网络类型 分类/检测(ssd/yolo)

  // Could possibly improve choice of exponent. Experiments show LeNet needs
  // 4bits, but the saturation border is at 3bits (when assuming infinitely long
  // mantisssa).
  this->exp_bits_ = 4;// 指数位大小
}

// 执行量化================================
void Quantization::QuantizeNet() {
  CheckWritePermissions(model_quantized_);//量化后的  prototxt 框架文件 写权限
  SetGpu();// 设置gpu模式
  // Run the reference floating point network on validation set to find baseline
  // accuracy.
  
// 首先 获取原有网络准确度==========================================
  // 
  Net<float>* net_val = new Net<float>(model_, caffe::TEST);// 测试网络
  net_val->CopyTrainedLayersFrom(weights_);//载入原来模型 对应的 模型权重
  float accuracy;
  // 获取指定网络的前向传播精度  这里需要区分 分了和检测网络 可参考 solver.cpp 中的测试部分
  RunForwardBatches(this->net_type_, this->iterations_, net_val, &accuracy);
  test_score_baseline_ = accuracy;
  delete net_val;// 删除测试网络
  
  
  // Run the reference floating point network on train data set to find maximum
  // values. Do statistic for 10 batches.
  Net<float>* net_test = new Net<float>(model_, caffe::TRAIN);// 训练网络
  net_test->CopyTrainedLayersFrom(weights_);//载入原来模型 对应的 模型权重
  RunForwardBatches(this->net_type_, 10, net_test, &accuracy, true);
  delete net_test;// 删除 训练网络
  // Do network quantization and scoring.
  
  // 根据不同的量化策略对原网络进行量化
  if (trimming_mode_ == "dynamic_fixed_point") 
  {
    Quantize2DynamicFixedPoint();// 动态固定点
  } 
  else if (trimming_mode_ == "minifloat") 
  {
    Quantize2MiniFloat();// 迷你浮点 16=1+10+1   
  } 
  else if (trimming_mode_ == "integer_power_of_2_weights") 
  {
    Quantize2IntegerPowerOf2Weights();// 2方 多比特位
  } 
  else {
    LOG(FATAL) << "Unknown trimming mode: " << trimming_mode_;
  }
}

// 文件读写权限===========================================
void Quantization::CheckWritePermissions(const string path) {
  std::ofstream probe_ofs(path.c_str());
  if (probe_ofs.good()) {
    probe_ofs.close();
    std::remove(path.c_str());
  } else {
    LOG(FATAL) << "Missing write permissions";
  }
}
// GPU =================================
void Quantization::SetGpu() {
  // Parse GPU ids or use all available devices
  vector<int> gpus;
  if (gpus_ == "all") {// 所以可用的GPU
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus.push_back(i);
    }
  } else if (gpus_.size()) {
    vector<string> strings;
    boost::split(strings, gpus_, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus.push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus.size(), 0);
  }
  // Set device id and mode
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
}

// 获取指定网络的前向传播精度  这里需要区分 分了和检测网络 可参考 solver.cpp 中的测试部分
void Quantization::RunForwardBatches(const string& net_type, const int iterations,
      Net<float>* caffe_net, float* accuracy, const bool do_stats,
      const int score_number) {
 // EvaluateDetection(iterations, caffe_net, accuracy, do_stats, score_number);
 ////////////////////
//Test(test_net_id); // 分类网络测试 或者 检测网络测试
    if ( net_type == "classification" ) 
    {// 分类网络测试评估结果
      EvaluateClassification(iterations, caffe_net, accuracy, do_stats, score_number);
    } 
    else if ( net_type == "ssd_detection" ) 
    {// 检测网络测试评估结果
      EvaluateDetectionSSD(iterations, caffe_net, accuracy, do_stats, score_number);
    } 
    else 
    {
      LOG(FATAL) << "Unknown evaluation type: " << net_type ;
    }
///////////////////////////////
}

// 分类网络评估=====================================
void Quantization::EvaluateClassification(const int iterations, 
    Net<float>* caffe_net, float* accuracy, const bool do_stats,
    const int score_number) {

  LOG(INFO) << "Running for EvaluateClassification " << iterations << " iterations.";
  
  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < iterations; ++i) {
    
    float iter_loss;
    // Do forward propagation.
    const vector<Blob<float>*>& result =
        caffe_net->Forward(bottom_vec, &iter_loss);
    // Find maximal values in network.
    if(do_stats) {
      caffe_net->RangeInLayers(&layer_names_, &max_in_, &max_out_,
          &max_params_);
    }
/////////////////////////////
    LOG(INFO) << "iterations " << i << "  :" << iter_loss;
/////////////////////////////
    // Keep track of network score over multiple batches.
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net->blob_names()[
            caffe_net->output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net->blob_names()[
        caffe_net->output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net->blob_loss_weights()[
        caffe_net->output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }
  *accuracy = test_score[score_number] / iterations;

  }

// 检测网络评估 SSD ==========================================
// 参考 ssd/solver.cpp 中的TestDetection()
void Quantization::EvaluateDetectionSSD(const int iterations, 
    Net<float>* caffe_net, float* accuracy, const bool do_stats,
    const int score_number) {
  
  LOG(INFO) << "Running for EvaluateDetectionSSD " << iterations << " iterations.";
  
  //CHECK(Caffe::root_solver());
  //LOG(INFO) << "Iteration " << iter_
  //          << ", Testing net (#" << test_net_id << ")";
  //CHECK_NOTNULL(test_nets_[test_net_id].get())->
      //ShareTrainedLayersWith(net_.get());
  map<int, map<int, vector<pair<float, int> > > > all_true_pos;
  map<int, map<int, vector<pair<float, int> > > > all_false_pos;
  map<int, map<int, int> > all_num_pos;
  Net<float>*  test_net = caffe_net;
  float loss = 0;
  for (int i = 0; i < iterations; ++i) {
    float iter_loss;
    if(do_stats) {
      caffe_net->RangeInLayers(&layer_names_, &max_in_, &max_out_,
          &max_params_);
    }
    const vector<Blob<float>*>& result = test_net->Forward(&iter_loss);


////////////////////////
    LOG(INFO) << "iterations  " << i << "  :" << iter_loss;
/////////////////////


    loss += iter_loss;
    if (!do_stats) {
    for (int j = 0; j < result.size(); ++j) {
      // 目标检测结果维度
      CHECK_EQ(result[j]->width(), 5);//
      const float* result_vec = result[j]->cpu_data();
      // 总的检测结果数量
      int num_det = result[j]->height();
      // 遍历每一个检测结果
      for (int k = 0; k < num_det; ++k) {
        int item_id = static_cast<int>(result_vec[k * 5]);// id    
        int label = static_cast<int>(result_vec[k * 5 + 1]);// 标签 label
        if (item_id == -1) {
          // Special row of storing number of positives for a label.
          if (all_num_pos[j].find(label) == all_num_pos[j].end()) {
            all_num_pos[j][label] = static_cast<int>(result_vec[k * 5 + 2]);// 
          } else {
            all_num_pos[j][label] += static_cast<int>(result_vec[k * 5 + 2]);
          }
        } else {
          // Normal row storing detection status.
          float score = result_vec[k * 5 + 2];// 得分
          int tp = static_cast<int>(result_vec[k * 5 + 3]);//正确的位置预测 id
          int fp = static_cast<int>(result_vec[k * 5 + 4]);//错误的预测id
          if (tp == 0 && fp == 0) {
            // Ignore such case. It happens when a detection bbox is matched to
            // a difficult gt bbox and we don't evaluate on difficult gt bbox.
            continue;
          }
          all_true_pos[j][label].push_back(std::make_pair(score, tp));
          all_false_pos[j][label].push_back(std::make_pair(score, fp));
        }
      }
    }
    }
  }

    loss /= iterations;
    LOG(INFO) << "Dection test loss: " << loss;

  float mAP = 0.;
  if (!do_stats) {
  for (int i = 0; i < all_true_pos.size(); ++i) {
    if (all_true_pos.find(i) == all_true_pos.end()) {
      LOG(FATAL) << "Missing output_blob true_pos: " << i;
    }
    const map<int, vector<pair<float, int> > >& true_pos =
        all_true_pos.find(i)->second;
    if (all_false_pos.find(i) == all_false_pos.end()) {
      LOG(FATAL) << "Missing output_blob false_pos: " << i;
    }
    const map<int, vector<pair<float, int> > >& false_pos =
        all_false_pos.find(i)->second;
    if (all_num_pos.find(i) == all_num_pos.end()) {
      LOG(FATAL) << "Missing output_blob num_pos: " << i;
    }
    const map<int, int>& num_pos = all_num_pos.find(i)->second;
    map<int, float> APs;
    
    // Sort true_pos and false_pos with descend scores.
    for (map<int, int>::const_iterator it = num_pos.begin();
         it != num_pos.end(); ++it) {
      int label = it->first;
      int label_num_pos = it->second;
      if (true_pos.find(label) == true_pos.end()) {
        LOG(WARNING) << "Missing true_pos for label: " << label;
        continue;
      }
      const vector<pair<float, int> >& label_true_pos =
          true_pos.find(label)->second;
      if (false_pos.find(label) == false_pos.end()) {
        LOG(WARNING) << "Missing false_pos for label: " << label;
        continue;
      }
      const vector<pair<float, int> >& label_false_pos =
          false_pos.find(label)->second;
      vector<float> prec, rec;
      caffe::ComputeAP(label_true_pos, label_num_pos, label_false_pos,
                "11point", &prec, &rec, &(APs[label]));
      mAP += APs[label];
    }
    mAP /= num_pos.size();
 //   const int output_blob_index = test_net->output_blob_indices()[i];
 //   const string& output_name = test_net->blob_names()[output_blob_index];
 //   LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
 //             << mAP;
   }
  }
  LOG(INFO) << "accuracy : " << mAP << "mAP";
  *accuracy = mAP;
}

///////////////////////////////////////////////////////////////////////
// 检测网络评估 YOLOV2 ==========================================
// 参考 ssd/solver.cpp 中的TestDetection()
void Quantization::EvaluateDetectionYOLOV2(const int iterations, 
    Net<float>* caffe_net, float* accuracy, const bool do_stats,
    const int score_number) {
  
  LOG(INFO) << "Running for EvaluateDetectionYOLOV2 " << iterations << " iterations.";
  

  
  Net<float>*  test_net = caffe_net;
  
  float loss = 0;
  float mAP = 0.;
  for (int i = 0; i < iterations; ++i) {
	  
    float iter_loss;
	
	///   获取网络各层参数范围
    if(do_stats) 
	{
      caffe_net->RangeInLayers(&layer_names_, &max_in_, &max_out_,
          &max_params_);
    }
	
    test_net->Forward(&iter_loss);
	const shared_ptr<Blob<float> > result_ptr = test_net->blob_by_name("eval_det"); // 检测评估层输出
	const float* pstart = result_ptr->cpu_data();

///
    LOG(INFO) << "iterations  " << i << "  :" << iter_loss;
///
	loss += iter_loss;// loss 之和
	mAP += *pstart; // eval_detection_layer 那边已经修改成 获取整个batch 的mAP
    }
	if(iterations) loss /= iterations;
	if(iterations) mAP /= iterations;
	LOG(INFO) << "Dection test loss: " << loss;
	LOG(INFO) << "accuracy : " << mAP << "mAP";
	*accuracy = mAP;

}
////////////////////////////////////////////////////


/*
void Quantization::RunForwardBatches(const int iterations,
      Net<float>* caffe_net, float* accuracy, const bool do_stats,
      const int score_number) {
  LOG(INFO) << "Running for " << iterations << " iterations.";
  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  
  float loss = 0;
  for (int i = 0; i < iterations; ++i) {
    float iter_loss;
    vector<Blob<float>*> result;
    // Do forward propagation.
    if (!do_stats) {
      //const vector<Blob<float>*>& result1 =
          //caffe_net->Forward(bottom_vec, &iter_loss);

      float mAP;
      //const vector<Blob<float>*> result;
      Blob<float>* blob_input_(new Blob<float>());
      result.push_back(blob_input_);
      vector<int> top_shape(0);
      result[0]->Reshape(top_shape);
      //mAP =  EvaluateDetection(result1);
	mAP =  EvaluateDetection(caffe_net, );
      result[0]->mutable_cpu_data()[0] = mAP;
    }
    else {
      //const vector<Blob<float>*>& result = caffe_net->Forward(bottom_vec, &iter_loss);
      result = caffe_net->Forward(bottom_vec, &iter_loss);
    }

    // Find maximal values in network.
    if(do_stats) {
      caffe_net->RangeInLayers(&layer_names_, &max_in_, &max_out_,
          &max_params_);
    }
    // Keep track of network score over multiple batches.
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net->blob_names()[
            caffe_net->output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net->blob_names()[
        caffe_net->output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net->blob_loss_weights()[
        caffe_net->output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }
  *accuracy = test_score[score_number] / iterations;
}
*/


/////////////////////// 动态固定点=======================================
void Quantization::Quantize2DynamicFixedPoint() {
  // Find the integer length for dynamic fixed point numbers.
  // The integer length is chosen such that no saturation occurs.
  // This approximation assumes an infinitely long factional part.
  // For layer activations, we reduce the integer length by one bit.
  for (int i = 0; i < layer_names_.size(); ++i) {
    il_in_.push_back((int)ceil(log2(max_in_[i])));
    il_out_.push_back((int)ceil(log2(max_out_[i])));
    il_params_.push_back((int)ceil(log2(max_params_[i])+1));
  }
  // Debug
  for (int k = 0; k < layer_names_.size(); ++k) {
    LOG(INFO) << "Layer " << layer_names_[k] <<
        ", integer length input=" << il_in_[k] <<
        ", integer length output=" << il_out_[k] <<
        ", integer length parameters=" << il_params_[k];
  }

  // Score net with dynamic fixed point convolution parameters.
  // The rest of the net remains in high precision format.
  NetParameter param;
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  vector<int> test_bw_conv_params;
  vector<float> test_scores_conv_params;
  float accuracy;
  Net<float>* net_test;
  for (int bitwidth = 16; bitwidth > 4; bitwidth /= 2) {
    EditNetDescriptionDynamicFixedPoint(&param, "Convolution", "Parameters",
        bitwidth, -1, -1, -1);
    net_test = new Net<float>(param, NULL);
    net_test->CopyTrainedLayersFrom(weights_);
    RunForwardBatches(this->net_type_, iterations_, net_test, &accuracy);
    test_bw_conv_params.push_back(bitwidth);
    test_scores_conv_params.push_back(accuracy);
    delete net_test;
    //if ( accuracy + error_margin_ / 100 < test_score_baseline_ ) break;
  }

  // Score net with dynamic fixed point inner product parameters.
  // The rest of the net remains in high precision format.
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  vector<int> test_bw_fc_params;
  vector<float> test_scores_fc_params;
  for (int bitwidth = 8; bitwidth > 4; bitwidth /= 2) {
    EditNetDescriptionDynamicFixedPoint(&param, "InnerProduct", "Parameters",
        -1, bitwidth, -1, -1);
    net_test = new Net<float>(param, NULL);
    net_test->CopyTrainedLayersFrom(weights_);
    RunForwardBatches(this->net_type_, iterations_, net_test, &accuracy);
    test_bw_fc_params.push_back(bitwidth);
    test_scores_fc_params.push_back(accuracy);
    delete net_test;
    //if ( accuracy + error_margin_ / 100 < test_score_baseline_ ) break;
  }

  // Score net with dynamic fixed point layer activations.
  // The rest of the net remains in high precision format.
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  vector<int> test_bw_layer_activations;
  vector<float> test_scores_layer_activations;
  for (int bitwidth = 16; bitwidth > 4; bitwidth /= 2) {
    EditNetDescriptionDynamicFixedPoint(&param, "Convolution_and_InnerProduct",
        "Activations", -1, -1, bitwidth, bitwidth);
    net_test = new Net<float>(param, NULL);
    net_test->CopyTrainedLayersFrom(weights_);
    RunForwardBatches(this->net_type_, iterations_, net_test, &accuracy);
    test_bw_layer_activations.push_back(bitwidth);
    test_scores_layer_activations.push_back(accuracy);
    delete net_test;
    if ( accuracy + error_margin_ / 100 < test_score_baseline_ ) break;
  }

  // Choose bit-width for different network parts
  bw_conv_params_ = 32;
  bw_fc_params_ = 32;
  bw_out_ = 32;
  for (int i = 0; i < test_scores_conv_params.size(); ++i) {
    if (test_scores_conv_params[i] + error_margin_ / 100 >=
          test_score_baseline_)
      bw_conv_params_ = test_bw_conv_params[i];
    else
      break;
  }
  for (int i = 0; i < test_scores_fc_params.size(); ++i) {
    if (test_scores_fc_params[i] + error_margin_ / 100 >=
          test_score_baseline_)
      bw_fc_params_ = test_bw_fc_params[i];
    else
      break;
  }
  for (int i = 0; i < test_scores_layer_activations.size(); ++i) {
    if (test_scores_layer_activations[i] + error_margin_ / 100 >=
          test_score_baseline_)
      bw_out_ = test_bw_layer_activations[i];
    else
      break;
  }
  bw_in_ = bw_out_;

  // Score dynamic fixed point network.
  // This network combines dynamic fixed point parameters in convolutional and
  // inner product layers, as well as dynamic fixed point activations.
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  EditNetDescriptionDynamicFixedPoint(&param, "Convolution_and_InnerProduct",
      "Parameters_and_Activations", bw_conv_params_, bw_fc_params_, bw_in_,
      bw_out_);
  net_test = new Net<float>(param, NULL);
  net_test->CopyTrainedLayersFrom(weights_);
  RunForwardBatches(this->net_type_, iterations_, net_test, &accuracy);
  delete net_test;
  param.release_state();
  WriteProtoToTextFile(param, model_quantized_);

  // Write summary of dynamic fixed point analysis to log
  LOG(INFO) << "------------------------------";
  LOG(INFO) << "Network accuracy analysis for";
  LOG(INFO) << "Convolutional (CONV) and fully";
  LOG(INFO) << "connected (FC) layers.";
  LOG(INFO) << "Baseline 32bit float: " << test_score_baseline_;
  LOG(INFO) << "Dynamic fixed point CONV";
  LOG(INFO)  << "weights: ";
  for (int j = 0; j < test_scores_conv_params.size(); ++j) {
    LOG(INFO) << test_bw_conv_params[j] << "bit: \t" <<
        test_scores_conv_params[j];
  }
  LOG(INFO) << "Dynamic fixed point FC";
  LOG(INFO) << "weights: ";
  for (int j = 0; j < test_scores_fc_params.size(); ++j) {
    LOG(INFO) << test_bw_fc_params[j] << "bit: \t" << test_scores_fc_params[j];
  }
  LOG(INFO) << "Dynamic fixed point layer";
  LOG(INFO) << "activations:";
  for (int j = 0; j < test_scores_layer_activations.size(); ++j) {
    LOG(INFO) << test_bw_layer_activations[j] << "bit: \t" <<
        test_scores_layer_activations[j];
  }
  LOG(INFO) << "Dynamic fixed point net:";
  LOG(INFO) << bw_conv_params_ << "bit CONV weights,";
  LOG(INFO) << bw_fc_params_ << "bit FC weights,";
  LOG(INFO) << bw_out_ << "bit layer activations:";
  LOG(INFO) << "Accuracy: " << accuracy;
  LOG(INFO) << "Please fine-tune.";
}

////////////////////////////////// 迷你浮点 16=1+10+1  =====================================
void Quantization::Quantize2MiniFloat() {
  // Find the necessary amount of exponent bits.
  // The exponent bits are chosen such that no saturation occurs.
  // This approximation assumes an infinitely long mantissa.
  // Parameters are ignored, since they are normally smaller than layer
  // activations.
  for ( int i = 0; i < layer_names_.size(); ++i ) {
    int exp_in = ceil(log2(log2(max_in_[i]) - 1) + 1);
    int exp_out = ceil(log2(log2(max_out_[i]) - 1) + 1);
    exp_bits_ = std::max( std::max( exp_bits_, exp_in ), exp_out);
  }

  // Score net with minifloat parameters and activations.
  NetParameter param;
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  vector<int> test_bitwidth;
  vector<float> test_scores;
  float accuracy;
  Net<float>* net_test;
  // Test the net with different bit-widths
  for (int bitwidth = 16; bitwidth - 1 - exp_bits_ > 0; bitwidth /= 2) {
    EditNetDescriptionMiniFloat(&param, bitwidth);
    net_test = new Net<float>(param, NULL);
    net_test->CopyTrainedLayersFrom(weights_);
    RunForwardBatches(this->net_type_, iterations_, net_test, &accuracy);
    test_bitwidth.push_back(bitwidth);
    test_scores.push_back(accuracy);
    delete net_test;
    if ( accuracy + error_margin_ / 100 < test_score_baseline_ ) break;
  }

  // Choose bitwidth for network
  int best_bitwidth = 32;
  for(int i = 0; i < test_scores.size(); ++i) {
    if (test_scores[i] + error_margin_ / 100 >= test_score_baseline_)
      best_bitwidth = test_bitwidth[i];
    else
      break;
  }

  // Write prototxt file of net with best bitwidth
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  EditNetDescriptionMiniFloat(&param, best_bitwidth);
  WriteProtoToTextFile(param, model_quantized_);

  // Write summary of minifloat analysis to log
  LOG(INFO) << "------------------------------";
  LOG(INFO) << "Network accuracy analysis for";
  LOG(INFO) << "Convolutional (CONV) and fully";
  LOG(INFO) << "connected (FC) layers.";
  LOG(INFO) << "Baseline 32bit float: " << test_score_baseline_;
  LOG(INFO) << "Minifloat net:";
  for(int j = 0; j < test_scores.size(); ++j) {
    LOG(INFO) << test_bitwidth[j] << "bit: \t" << test_scores[j];
  }
  LOG(INFO) << "Please fine-tune.";
}


/////////////////// 2方 多比特位====================================
void Quantization::Quantize2IntegerPowerOf2Weights() {
  // Find the integer length for dynamic fixed point numbers.
  // The integer length is chosen such that no saturation occurs.
  // This approximation assumes an infinitely long factional part.
  // For layer activations, we reduce the integer length by one bit.
  for (int i = 0; i < layer_names_.size(); ++i) {
    il_in_.push_back((int)ceil(log2(max_in_[i])));
    il_out_.push_back((int)ceil(log2(max_out_[i])));
  }
  // Score net with integer-power-of-two weights and dynamic fixed point
  // activations.
  NetParameter param;
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  float accuracy;
  Net<float>* net_test;
  EditNetDescriptionIntegerPowerOf2Weights(&param);
  // Bit-width of layer activations is hard-coded to 8-bit.
  EditNetDescriptionDynamicFixedPoint(&param, "Convolution_and_InnerProduct",
      "Activations", -1, -1, 8, 8);
  net_test = new Net<float>(param, NULL);
  net_test->CopyTrainedLayersFrom(weights_);
  RunForwardBatches(this->net_type_, iterations_, net_test, &accuracy);
  delete net_test;

  // Write prototxt file of quantized net
  param.release_state();
  WriteProtoToTextFile(param, model_quantized_);

  // Write summary of integer-power-of-2-weights analysis to log
  LOG(INFO) << "------------------------------";
  LOG(INFO) << "Network accuracy analysis for";
  LOG(INFO) << "Integer-power-of-two weights";
  LOG(INFO) << "in Convolutional (CONV) and";
  LOG(INFO) << "fully connected (FC) layers.";
  LOG(INFO) << "Baseline 32bit float: " << test_score_baseline_;
  LOG(INFO) << "Quantized net:";
  LOG(INFO) << "4bit: \t" << accuracy;
  LOG(INFO) << "Please fine-tune.";
}

///////////////////////////////////// 
void Quantization::EditNetDescriptionDynamicFixedPoint(NetParameter* param,
      const string layers_2_quantize, const string net_part, const int bw_conv,
      const int bw_fc, const int bw_in, const int bw_out) {
  for (int i = 0; i < param->layer_size(); ++i) {
    // if this is a convolutional layer which should be quantized ...
    if (layers_2_quantize.find("Convolution") != string::npos &&
        param->layer(i).type().find("Convolution") != string::npos) {
      // quantize parameters  量化卷积参数
      if (net_part.find("Parameters") != string::npos) {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("ConvolutionRistretto");
        param_layer->mutable_quantization_param()->set_fl_params(bw_conv -
            GetIntegerLengthParams(param->layer(i).name()));
        param_layer->mutable_quantization_param()->set_bw_params(bw_conv);
      }
      // quantize activations 量化激活层
      if (net_part.find("Activations") != string::npos) {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("ConvolutionRistretto");
        param_layer->mutable_quantization_param()->set_fl_layer_in(bw_in -
            GetIntegerLengthIn(param->layer(i).name()));
        param_layer->mutable_quantization_param()->set_bw_layer_in(bw_in);
        param_layer->mutable_quantization_param()->set_fl_layer_out(bw_out -
            GetIntegerLengthOut(param->layer(i).name()));
        param_layer->mutable_quantization_param()->set_bw_layer_out(bw_out);
      }
    }
    // if this is an inner product layer which should be quantized ...
    if (layers_2_quantize.find("InnerProduct") != string::npos &&
        (param->layer(i).type().find("InnerProduct") != string::npos ||
        param->layer(i).type().find("FcRistretto") != string::npos)) {
      // quantize parameters 量化全链接参数
      if (net_part.find("Parameters") != string::npos) {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("FcRistretto");
        param_layer->mutable_quantization_param()->set_fl_params(bw_fc -
            GetIntegerLengthParams(param->layer(i).name()));
        param_layer->mutable_quantization_param()->set_bw_params(bw_fc);
      }
      // quantize activations  量化激活层
      if (net_part.find("Activations") != string::npos) {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("FcRistretto");
        param_layer->mutable_quantization_param()->set_fl_layer_in(bw_in -
            GetIntegerLengthIn(param->layer(i).name()) );
        param_layer->mutable_quantization_param()->set_bw_layer_in(bw_in);
        param_layer->mutable_quantization_param()->set_fl_layer_out(bw_out -
            GetIntegerLengthOut(param->layer(i).name()) );
        param_layer->mutable_quantization_param()->set_bw_layer_out(bw_out);
      }
    }
  }
}

void Quantization::EditNetDescriptionMiniFloat(NetParameter* param,
      const int bitwidth) {
  caffe::QuantizationParameter_Precision precision =
        caffe::QuantizationParameter_Precision_MINIFLOAT;
  for (int i = 0; i < param->layer_size(); ++i) {
    if ( param->layer(i).type() == "Convolution" ||
          param->layer(i).type() == "ConvolutionRistretto") {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("ConvolutionRistretto");
      param_layer->mutable_quantization_param()->set_precision(precision);
      param_layer->mutable_quantization_param()->set_mant_bits(bitwidth
          - exp_bits_ - 1);
      param_layer->mutable_quantization_param()->set_exp_bits(exp_bits_);
    } else if ( param->layer(i).type() == "InnerProduct" ||
          param->layer(i).type() == "FcRistretto") {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("FcRistretto");
      param_layer->mutable_quantization_param()->set_precision(precision);
      param_layer->mutable_quantization_param()->set_mant_bits(bitwidth
          - exp_bits_ - 1);
      param_layer->mutable_quantization_param()->set_exp_bits(exp_bits_);
    }
  }
}

void Quantization::EditNetDescriptionIntegerPowerOf2Weights(
      NetParameter* param) {
  caffe::QuantizationParameter_Precision precision =
      caffe::QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS;
  for (int i = 0; i < param->layer_size(); ++i) {
    if ( param->layer(i).type() == "Convolution" ||
          param->layer(i).type() == "ConvolutionRistretto") {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("ConvolutionRistretto");
      param_layer->mutable_quantization_param()->set_precision(precision);
      // Weights are represented as 2^e where e in [-8,...,-1].
      // This choice of exponents works well for AlexNet.
      param_layer->mutable_quantization_param()->set_exp_min(-8);
      param_layer->mutable_quantization_param()->set_exp_max(-1);
    } else if ( param->layer(i).type() == "InnerProduct" ||
          param->layer(i).type() == "FcRistretto") {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("FcRistretto");
      param_layer->mutable_quantization_param()->set_precision(precision);
      // Weights are represented as 2^e where e in [-8,...,-1].
      // This choice of exponents works well for AlexNet.
      param_layer->mutable_quantization_param()->set_exp_min(-8);
      param_layer->mutable_quantization_param()->set_exp_max(-1);
    }
  }
}

int Quantization::GetIntegerLengthParams(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return il_params_[pos];
}

int Quantization::GetIntegerLengthIn(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return il_in_[pos];
}

int Quantization::GetIntegerLengthOut(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return il_out_[pos];
}
