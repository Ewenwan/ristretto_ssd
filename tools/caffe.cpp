// caffe 工具入口  训练测试模型
#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp" // 包含各种头文件　blob，layer，net ,  solver 等
#include "caffe/util/signal_handler.h"

///////////////////////////////////////
#include <map>
#include <utility>
#include "caffe/util/bbox_util.hpp"// ComputeAP()
using namespace std;
////////////////////////////////////////////



using caffe::Blob; // 作为数据传输的媒介，无论是网络权重参数，还是输入数据，都是转化为Blob数据结构来存储
// 可以把Blob看成一个有4维的结构体（包含数据和梯度），而实际上，它们只是一维的指针而已，其4维结构通过shape属性得以计算出来。
// shared_ptr<SyncedMemory> data_ //数据
// shared_ptr<SyncedMemory> diff_ //梯度
// void Blob<Dtype>::Reshape(const int num, const int channels, const int height,const int width)
// 在更高一级的Layer中Blob用下面的形式表示学习到的参数：
// vector<shared_ptr<Blob<Dtype> > > blobs_;
// 这里使用的是一个Blob的容器是因为某些Layer包含多组学习参数，比如多个卷积核的卷积层。
// 以及Layer所传递的数据形式，后面还会涉及到这里：
// vector<Blob<Dtype>*> &bottom;
// vector<Blob<Dtype>*> *top

using caffe::Layer;// 作为网络的基础单元，神经网络中层与层间的数据节点、前后传递都在该数据结构中被实现，
// 层类种类丰富，比如常用的卷积层、全连接层、pooling层等等，大大地增加了网络的多样性.
// NeuronLayer类 定义于neuron_layers.hpp中, 比如Dropout运算，激活函数ReLu，Sigmoid等.
// LossLayer类 定义于loss_layers.hpp中，其派生类会产生loss，只有这些层能够产生loss。
// 数据层 定义于data_layer.hpp中，作为网络的最底层，主要实现数据格式的转换。
// 特征表达层（我自己分的类）定义于vision_layers.hpp, 包含卷积操作，Pooling操作.
// 网络连接层和激活函数（我自己分的类）定义于common_layers.hpp，包括了常用的 全连接层 InnerProductLayer 类。

// 在Layer内部，数据主要有两种传递方式，正向传导（Forward）和反向传导（Backward）。Forward和Backward有CPU和GPU（部分有）两种实现。
// virtual void Forward(const vector<Blob<Dtype>*> &bottom, vector<Blob<Dtype>*> *top) = 0;
// virtual void Backward(const vector<Blob<Dtype>*> &top, const vector<bool> &propagate_down,  vector<Blob<Dtype>*> *bottom) = 0;
// Layer类派生出来的层类通过这实现这两个虚函数，产生了各式各样功能的层类。
// Forward是从根据bottom计算top的过程，Backward则相反（根据top计算bottom）。

using caffe::Caffe;
using caffe::Net;// 作为网络的整体骨架，决定了网络中的层次数目以及各个层的类别等信息
// Net用容器的形式将多个Layer有序地放在一起，其自身实现的功能主要是对逐层Layer进行初始化，
// 以及提供Update( )的接口（更新网络参数），本身不能对参数进行有效地学习过程。
// vector<shared_ptr<Layer<Dtype> > > layers_;
// 同样Net也有它自己的 前向传播 和反向传播
// vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom, Dtype* loss = NULL); // 前传得到　loss
// void Net<Dtype>::Backward();// 反传播得到各个层参数的梯度
// 他们是对整个网络的前向和方向传导，各调用一次就可以计算出网络的loss了。


using caffe::Solver;// 作为网络的求解策略，涉及到求解优化问题的策略选择以及参数确定方面，修改这个模块的话一般都会是研究DL的优化求解的方向。
// 包含一个Net的指针，主要是实现了训练模型参数所采用的优化算法，它所派生的类就可以对整个网络进行训练了。
// shared_ptr<Net<Dtype> > net_;
// 不同的模型训练方法通过重载函数ComputeUpdateValue( )实现计算update参数的核心功能.
// virtual void ComputeUpdateValue() = 0;
// 进行整个网络训练过程（也就是你运行Caffe训练某个模型）的时候，实际上是在运行caffe.cpp中的train( )函数，
// 而这个函数实际上是实例化一个Solver对象，初始化后调用了Solver中的Solve( )方法。
// 而这个Solve( )函数主要就是在迭代运行下面这两个函数，就是刚才介绍的哪几个函数。
// ComputeUpdateValue();
// net_->Update();

using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

//首先需要 使用 gflags的宏：DEFINE_xxxxx(变量名，默认值，help-string)  定义命令行参数==================
// 使用 google::ParseCommandLineFlags(&argc, &argv, true); 解析后
// 就可以使用 FLAGS_ 变量名访问对应的命令行参数了
// 例如 FLAGS_gpu  FLAGS_solver FLAGS_model  FLAGS_snapshot  FLAGS_weights  FLAGS_iterations

// gflags.h 参数 gpu序列
DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
// 求解器prototxt文件名
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
// 网络模型框架文件名
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
// 断点继续训练的　模型权重状态文件
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
// 网络模型权重 文件名
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
//迭代次数
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
// 停止标志信息
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
// 模型保存地址  断点训练
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// ========================== add net type ================
// 分类网络/检测网络(classification /ssd_detection / yolov2_detection / faster_rcnn_detection)
DEFINE_string(net_type, "", "The net type: classification /ssd_detection / yolov2_detection");
/////////////////////////////////////////


// 函数 注册函数
// A simple registry for caffe commands.
typedef int (*BrewFunction)();// 函数指针
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;//g_brew_map 函数名与函数指针键值对map映射

//　不同函数的注册类 将函数 记录到 g_brew_map 
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

// 由函数名字符串获取 执行对应应函数指针的函数==========================
static BrewFunction GetBrewFunction(const caffe::string& name) {
      if (g_brew_map.count(name)) //查看记录的键值对映射中是否有该种类型的层名字
      {
	  return g_brew_map[name];// 返回对应函数名字的函数指针 并执行相应的函数
      }
      else 
      {
	  LOG(ERROR) << "Available caffe actions:";
	  for (BrewMap::iterator it = g_brew_map.begin(); it != g_brew_map.end(); ++it) 
	  {
	    LOG(ERROR) << "\t" << it->first;// 打印支持的层名字
	  }
	  LOG(FATAL) << "Unknown action: " << name;//　打印对应的不存在的 层名字
	  return NULL;  // not reachable, just to suppress old compiler warnings.
      }
}

// 从命令行参数 FLAGS_gpu 中, 得到使用GPU的id序列 
static void get_gpus(vector<int>* gpus)
{
      
      if (FLAGS_gpu == "all") // 使用所有可用的GPU
      {
	  int count = 0;
      #ifndef CPU_ONLY
	  CUDA_CHECK(cudaGetDeviceCount(&count));//　获取可用的GPU数量
      #else
	  NO_GPU;
      #endif
	  for (int i = 0; i < count; ++i) 
	  {
	    gpus->push_back(i);//依次加入gpus GPU的id序列容器vector中
	  }
      }
      else if (FLAGS_gpu.size()) // GPU序列字符串长度大于0
      {
	  vector<string> strings;// 命令行指定的GPU序列字符串
	  boost::split(strings, FLAGS_gpu, boost::is_any_of(","));// 按逗号"," 分割GPU序列字符串
	  for (int i = 0; i < strings.size(); ++i) {
	    // lexical_cast c++数据类型万能转换器  string 转换到 int 
	    gpus->push_back(boost::lexical_cast<int>(strings[i]));// 转换类型后，加入到 gpus GPU的id序列容器vector中
	  }
      } 
      else 
      {
	  CHECK_EQ(gpus->size(), 0);// 未指定需要使用的GPU序列 
      }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
// 从命令行参数 FLAGS_gpu 中,得到使用GPU的id序列
// 并设置使用对应的GPU设备
int device_query()
{
      LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
      vector<int> gpus;// 需要使用的GPUid序列容器vector
      get_gpus(&gpus);// 使用 get_gpus 从命令行参数 FLAGS_gpu 中　得到使用GPU的id序列
      for (int i = 0; i < gpus.size(); ++i) {// 遍历gpus id序列
	caffe::Caffe::SetDevice(gpus[i]);//设置使用对应的GPU设备
	caffe::Caffe::DeviceQuery();
      }
      
      return 0;
}
// 注册该函数到 g_brew_map (函数名与函数指针键值对map映射)
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
// 从 模型文件 model_list 导入参数到 
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list)
{
    std::vector<std::string> model_names;
    boost::split(model_names, model_list, boost::is_any_of(",") );// 按逗号","分割模型权重文件　
    for (int i = 0; i < model_names.size(); ++i) // 多权重文件
    {
	LOG(INFO) << "Finetuning from " << model_names[i];// 模型权重文件名
	
	solver->net()->CopyTrainedLayersFrom(model_names[i]);// 拷贝训练网络参数权重
	
	for (int j = 0; j < solver->test_nets().size(); ++j)// 拷贝测试网络参数权重
	{
	    solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
	}
    }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction( const std::string& flag_value) 
{
      if (flag_value == "stop") {
	return caffe::SolverAction::STOP;
      }
      if (flag_value == "snapshot") {// 模型保存地址
	return caffe::SolverAction::SNAPSHOT;
      }
      if (flag_value == "none") {
	return caffe::SolverAction::NONE;
      }
      LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

// 训练或者微调一个网络
// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())// 断点训练　/ 微调
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;// 求解器参数
  // 从解析器文件solver.prototxt文件读取　求解器参数　solver_param===================
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

  // 优先从命令行　获取GPU　序列字符串==============================
  // 其次从解析器文件solver.prototxt文件读取gpu id序列字符串
  // 否则默认为 0号
  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {// 求解析器文件内定义的 gpu id序列
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);// 如果命令行和求解析文件内定义均为定义gpu,则默认为0号GPU
      }
  }
// 从gpu id序列字符串中获取 gpu id序列容器==============================
  vector<int> gpus;
  get_gpus(&gpus);
  
  if (gpus.size() == 0)
  {
      LOG(INFO) << "Use CPU.";
      Caffe::set_mode(Caffe::CPU);//设置为CPU模式==============================
  } 
  else 
  {
	ostringstream s;
	for (int i = 0; i < gpus.size(); ++i) {
	  s << (i ? ", " : "") << gpus[i];
	}
	LOG(INFO) << "Using GPUs " << s.str();
    #ifndef CPU_ONLY
	cudaDeviceProp device_prop;
	for (int i = 0; i < gpus.size(); ++i) {
	  cudaGetDeviceProperties(&device_prop, gpus[i]);// gpu设备优先权
	  LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
	}
    #endif
	solver_param.set_device_id(gpus[0]);
	Caffe::SetDevice(gpus[0]);
	Caffe::set_mode(Caffe::GPU);//设置为GPU模式==============================
	Caffe::set_solver_count(gpus.size());
  }
 
// 求解器相关　==============================
  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());
// 从...恢复上次的求解器信息==============================
  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  }
  // 拷贝模型权重==============================
  else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }
  
// 多gpu并行训练==============================
  if (gpus.size() > 1) {
    caffe::P2PSync<float> sync(solver, NULL, solver->param());
    sync.Run(gpus);
  }
  else
  {
    LOG(INFO) << "Starting Optimization";
    solver->Solve();//　单gpu训练
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
// 注册该函数到 g_brew_map (函数名与函数指针键值对map映射)==============================
RegisterBrewFunction(train);


////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////
// 分类网络 测试==========================================================
//template <typename Dtype>
void TestClassification() {
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);// 模型
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);// 权重
  LOG(INFO) << "Test Classification net running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    // 网络前传====================
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
     
    loss += iter_loss;// loss累加
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];// 分类网络得分
        if (i == 0) 
        {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } 
        else 
        {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }
}


// 检测网络测试 SSD ======================================================
//template <typename Dtype>
void TestDetectionSSD() {
  // Instantiate the caffe net.
  //Net<float> caffe_net(FLAGS_model, caffe::TEST);// 模型
  Net<float>* caffe_net = new Net<float>(FLAGS_model, caffe::TEST);// 测试网络模型
  caffe_net->CopyTrainedLayersFrom(FLAGS_weights);// 权重
  LOG(INFO) << "Test SSD Detection net running for " << FLAGS_iterations << " iterations.";

  //SolverParameter param;
  
  map<int, map<int, vector<pair<float, int> > > > all_true_pos;
  map<int, map<int, vector<pair<float, int> > > > all_false_pos;
  map<int, map<int, int> > all_num_pos;
  
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {

    float iter_loss;
    const vector<Blob<float>*>& result = caffe_net->Forward(&iter_loss);
    
////////////////////////
    LOG(INFO) << "Forward loss iterations  " << i << "  :" << iter_loss;
/////////////////////

    //if (param_.test_compute_loss()) {
      loss += iter_loss;// loss求和
    //}
    // 
    for (int j = 0; j < result.size(); ++j) {
      CHECK_EQ(result[j]->width(), 5);// 目标检测结果维度
      const float* result_vec = result[j]->cpu_data();
      int num_det = result[j]->height();// 总的检测结果数量
      for (int k = 0; k < num_det; ++k) {
        int item_id = static_cast<int>(result_vec[k * 5]);// id 
        int label = static_cast<int>(result_vec[k * 5 + 1]);// 标签 label
        if (item_id == -1) {
          // Special row of storing number of positives for a label.
          if (all_num_pos[j].find(label) == all_num_pos[j].end()) {
            all_num_pos[j][label] = static_cast<int>(result_vec[k * 5 + 2]);
          } else {
            all_num_pos[j][label] += static_cast<int>(result_vec[k * 5 + 2]);
          }
        } else {
          // Normal row storing detection status.
          float score = result_vec[k * 5 + 2];// 得分
          int tp = static_cast<int>(result_vec[k * 5 + 3]);//正确的位置预测
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

  loss /= FLAGS_iterations;// loss 均值滤波
  LOG(INFO) << "Dection test loss: " << loss;
  
  // 分析 结果================================
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
    float mAP = 0.;
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
     // ComputeAP 在 bbox_util.hpp :453  
      caffe::ComputeAP(label_true_pos, label_num_pos, label_false_pos,
                "11point", &prec, &rec, &(APs[label]));
      mAP += APs[label];
      //if (param_.show_per_class_result()) {
      //  LOG(INFO) << "class" << label << ": " << APs[label];
      //}
    }
    // 打印精度
    mAP /= num_pos.size();// 计算均值 mAP
    const int output_blob_index = caffe_net->output_blob_indices()[i];
    const string& output_name = caffe_net->blob_names()[output_blob_index];
    LOG(INFO) << "    Test detection net output #" << i << ": " << output_name << " = "
              << mAP << "mAP";
  }
}
/////////////////////////////////////////////////////


// 检测网络测试 YOLOV2 ======================================================
//template <typename Dtype>
void TestDetectionYolov2() {
  // Instantiate the caffe net.
  Net<float>* caffe_net = new Net<float>(FLAGS_model, caffe::TEST);// 测试网络模型
  caffe_net->CopyTrainedLayersFrom(FLAGS_weights);// 权重
  LOG(INFO) << "Test Yolov2 Detection net running for " << FLAGS_iterations << " iterations.";

  float loss = 0;
  float mAP = 0.;
  
  for (int i = 0; i < FLAGS_iterations; ++i) 
	{
		float iter_loss;
		caffe_net->Forward(&iter_loss);
		const shared_ptr<Blob<float> > result_ptr = caffe_net->blob_by_name("eval_det"); // 检测评估层输出
		const float* pstart = result_ptr->cpu_data(); // eval_det->cpu_data()返回的是多维数据（数组）
		////////////////////////
		LOG(INFO) << "Forward loss iterations  " << i << "  :" << iter_loss;
		/////////////////////
		loss += iter_loss;// loss求和
		mAP += *pstart; // eval_detection_layer 那边已经修改成 获取整个batch 的mAP
	}
	loss /= FLAGS_iterations;// loss 均值
	LOG(INFO) << "Dection test loss: " << loss;
	// 打印精度
	mAP /= FLAGS_iterations;// 计算均值 mAP
	LOG(INFO) << "Test detection net output #" << ": " << mAP << "mAP";

}
/////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// 测试一个网络 分类 top5   检测mAP50
// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // 从gpu id序列字符串中获取 gpu id序列容器==============================
    // Set device id and mode
    vector<int> gpus;
    get_gpus(&gpus);
    if (gpus.size() != 0) 
    {
	LOG(INFO) << "Use GPU with device ID " << gpus[0];
    #ifndef CPU_ONLY
	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, gpus[0]);
	LOG(INFO) << "GPU device name: " << device_prop.name;
    #endif
	Caffe::SetDevice(gpus[0]);
	Caffe::set_mode(Caffe::GPU);//设置为GPU模式
    }
    else 
    {
	LOG(INFO) << "Use CPU.";
	Caffe::set_mode(Caffe::CPU);//设置为CPU模式
    }
  
  ////////////////////
//Test(test_net_id); // 分类网络测试 或者 检测网络测试
    if (FLAGS_net_type == "classification") {
      TestClassification();
    } else if (FLAGS_net_type == "ssd_detection") {
      TestDetection();
    } else {
      LOG(FATAL) << "Unknown evaluation type: " << FLAGS_net_type;
    }
///////////////////////////////

  return 0;
}
// 注册该函数到 g_brew_map (函数名与函数指针键值对map映射)==============================
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
// 模型执行的时间,记录并打印每一层的前传和反传时间=======================================================
int time() {
	CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
	
      // 从gpu id序列字符串中获取 gpu id序列容器==============================
	// Set device id and mode
	vector<int> gpus;
	get_gpus(&gpus);
	if (gpus.size() != 0) 
	{
	  LOG(INFO) << "Use GPU with device ID " << gpus[0];
	  Caffe::SetDevice(gpus[0]);
	  Caffe::set_mode(Caffe::GPU);//设置为GPU模式
	}
	else
	{
	  LOG(INFO) << "Use CPU.";
	  Caffe::set_mode(Caffe::CPU);//设置为CPU模式
	}
	
    // 初始化网络模型========================================================	
	// Instantiate the caffe net.
	Net<float> caffe_net(FLAGS_model, caffe::TRAIN);// 训练模式

	// Do a clean forward and backward pass, so that memory allocation are done
	// and future iterations will be more stable.
	LOG(INFO) << "Performing Forward";
	// Note that for the speed benchmark, we will assume that the network does
	// not take any input blobs.
	float initial_loss;
	// 网络前向传播 获得　损失
	caffe_net.Forward(&initial_loss);
	LOG(INFO) << "Initial loss: " << initial_loss;
	LOG(INFO) << "Performing Backward";
	// 网络反向传播 计算参数梯度
	caffe_net.Backward();

	const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
	const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();// 层输入
	const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();//　层输出
	const vector<vector<bool> >& bottom_need_backward = caffe_net.bottom_need_backward();// 需要反向传播更新参数的层 bool量
	LOG(INFO) << "*** Benchmark begins ***";
	LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
	Timer total_timer;
	total_timer.Start();// 总时间开始计时
	Timer forward_timer;// 前传时间
	Timer backward_timer;// 反传时间
	Timer timer;
	std::vector<double> forward_time_per_layer(layers.size(), 0.0);// 每一层前传时间
	std::vector<double> backward_time_per_layer(layers.size(), 0.0);// 每一层反传时间
	double forward_time = 0.0;
	double backward_time = 0.0;
	for (int j = 0; j < FLAGS_iterations; ++j) {
	  Timer iter_timer;// 每次迭代时间记录器
	  iter_timer.Start();// 开始计时
	  forward_timer.Start();// 前传计时开始
	  for (int i = 0; i < layers.size(); ++i) 
	  {
	    timer.Start();
	    layers[i]->Forward(bottom_vecs[i], top_vecs[i]);//每一层前传
	    forward_time_per_layer[i] += timer.MicroSeconds();// 记录时间
	  }
	  forward_time += forward_timer.MicroSeconds();
	  backward_timer.Start();// 反传开始计时
	  for (int i = layers.size() - 1; i >= 0; --i)
	  {
	    timer.Start();
	    layers[i]->Backward(top_vecs[i], bottom_need_backward[i],bottom_vecs[i]);//每一层反传
	    backward_time_per_layer[i] += timer.MicroSeconds();
	  }
	  backward_time += backward_timer.MicroSeconds();
	  LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
	    << iter_timer.MilliSeconds() << " ms.";
	}
	LOG(INFO) << "Average time per layer: ";
	for (int i = 0; i < layers.size(); ++i) {
	  // 打印每一层前传时间和反传时间=======================================
	  const caffe::string& layername = layers[i]->layer_param().name();
	  LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
	    "\tforward: " << forward_time_per_layer[i] / 1000 /
	    FLAGS_iterations << " ms.";
	    
	  LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
	    "\tbackward: " << backward_time_per_layer[i] / 1000 /
	    FLAGS_iterations << " ms.";
	}
	total_timer.Stop();
	LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
	  FLAGS_iterations << " ms.";
	LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
	  FLAGS_iterations << " ms.";
	LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
	  FLAGS_iterations << " ms.";
	LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
	LOG(INFO) << "*** Benchmark ends ***";
	
	return 0;
}
// 注册该函数到 g_brew_map (函数名与函数指针键值对map映射)==============================
RegisterBrewFunction(time);



int main(int argc, char** argv) {
  std::cout << "starting ..." << std::endl;
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  
// 定制你自己的help信息与version信息：(gflags里面已经定义了-h和--version，你可以通过以下方式定制它们的内容)	
  // Set version 版本信息
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  // 工具命令行参数
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();// 根据命令行参数执行响应函数
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
