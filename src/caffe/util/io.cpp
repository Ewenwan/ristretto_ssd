#include <fcntl.h>
/////////////////////////// 文件解析
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>// xml 的
// json的
/////////////////////////////
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {
/////////////////////////////add////////////
using namespace boost::property_tree;
//////////////////////////////
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

// 读取 prototxt 文件
bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}
// 写 prototxt 文件
void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}
// 读取二进制 prototxt 文件
bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}
//  写 二进制 prototxt 文件 
void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
///////////////////////////////////
// 返回原图片尺寸
// 读取 图片 成cvmat  检测 时 需要记录原始图片大小 转换标记标签
cv::Mat ReadImageToCVMat(
    const string& filename,// 图片
    const int height,      // 目标尺寸
	const int width, 
	const bool is_color,   // 彩色?
    int* ori_w,            // 原图像尺寸
	int* ori_h) 
	{
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  // 读取图像
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find imag file " << filename;
    return cv_img_origin;
  }
  // 记录原始图片尺寸大小 转换 边框标记值时 会用到
  *ori_w = cv_img_origin.cols;
  *ori_h = cv_img_origin.rows;
  if (height > 0 && width > 0) {
// 图片 变形 指定目标尺寸进行变形
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } 
  else 
  {
    cv_img = cv_img_origin;
  }
  return cv_img;
}



// 读取 图片 成cvmat   分类 时 不需要记录原始图片大小
cv::Mat ReadImageToCVMat(
const string& filename,// 图片
const int height,      // 目标尺寸
const int width, 
const bool is_color) 
{
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) 
  {// 图片 变形
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } 
  else 
  {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

// 分类数据转换===================================
// 图片+ 类别id  转换成 数据  Datum* datum
bool ReadImageToDatum(
const string& filename, // 图片路径
const int label,        // 类别 id 
const int height,       // 转换后的图像大小
const int width, 
const bool is_color,    // 彩色?
const std::string & encoding, //编码格式 
Datum* datum) // 转换后的数据
{
  // 读取 图片数据 考虑变形
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
		// 不编码  
        return ReadFileToDatum(filename, label, datum);
      // 编码
	  std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
	  
	  // 设置数据===============================
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));// 图像data 数据
      datum->set_label(label);// 图像对应的 类别id
      datum->set_encoded(true);// 编码标志
      return true;
    }
	// 不编码
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

// 检测数据转换 
// 图片+ 标注文件  转换成 数据  Datum* datum
bool ReadBoxDataToDatum(
	const string& filename, // 图片路径 
	const string& annoname, // xml标注文件路径
	const map<string, int>& label_map, // 标注文件 name标签 : class_id
	const int height,    // 尺寸变形 固定到网络输入大小
	const int width, 
	const bool is_color, // 彩色图像
	const std::string & encoding, // 编码格式 
	Datum* datum)        // 转换到的数据
{
  int ori_w, ori_h;// 原始图像大小  标注标签转换时用到
  // 读取图片数据  考虑变形
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color, &ori_w, &ori_h);
  if (cv_img.data) 
  {
    if (encoding.size()) 
	{
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
		// 不编码 按照文件格式读取图片
        return ReadFileToDatum(filename, annoname, label_map, ori_w, ori_h, datum);
      // 编码
	  std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
	  
	  // 设置图片 data 像素数据区域
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_encoded(true);// 编码标志
      // 读取xml格式的标签数据 在 datum中加入 边框标签 数据
	  // 读取 xml文件  size域获取图片尺寸
	  // object 域标注框数据 object.name 映射 name:id 获取真实 类别id
	  // object.bndbox 获取标注框(左下角、右上角坐标)
	  // 转换成 中心点坐标 和 变成尺寸 用图像尺寸归一化到0~1之间
      ParseXmlToDatum(annoname, label_map, ori_w, ori_h, datum);
      return true;
    }
	
	// 不编码 图片转成 datum
    CVMatToDatum(cv_img, datum);
    // read xml anno data
    ParseXmlToDatum(annoname, label_map, ori_w, ori_h, datum);
    return true;
  } 
  else 
  {
    return false;
  }
}




#endif  // USE_OPENCV

//////////////////////////////////////////////
// 标注文件 name标签 : class_id
int name_to_label(const string& name, const map<string, int>& label_map) {
  map<string, int>::const_iterator it = label_map.find(name);
  if (it == label_map.end()) 
    return -1;
  else
    return it->second;
}


// xml 格式数据转换成 datum=======================================
//  这里要区分是 xml/json/txt格式数据==========待修改
void ParseXmlToDatum(
const string& annoname, // xml 标注文件 
const map<string, int>& label_map,// 标注文件 name标签 : class_id
int ori_w, // 元图片尺寸
int ori_h, 
Datum* datum) 
{
/*
<annotation>
  <folder>val</folder>
  <filename>000000581781</filename>
  <source>
    <database>MS COCO 2017</database>
  </source>
  <size>
    <width>640</width>
    <height>478</height>   // 标注了图像大小
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  
  <object>                 // 目标物体边框
    <name>52</name>        // 这里的类别名字  与实际的 类别id 有一个映射关系
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>                // 边框 
      <xmin>136</xmin>      // 左下角  坐标
      <ymin>216</ymin>      
      <xmax>434</xmax>      // 右上角
      <ymax>316</ymax>
    </bndbox>
  </object>
</annotation>
*/
  ptree pt;// <boost/property_tree/ptree.hpp>
  //LOG(INFO) << " XML file annoname： " << annoname ;
  read_xml(annoname, pt);// <boost/property_tree/xml_parser.hpp>
  //LOG(INFO) << " read XML file  successful " ;
  
  int width(0), height(0);
  try 
  {// 标注文件记录的图片大小需要和 图像实际大小一致，避免标记数据错误
    height = pt.get<int>("annotation.size.height");
    width = pt.get<int>("annotation.size.width");
    CHECK_EQ(ori_w, width);
    CHECK_EQ(ori_h, height);
  } 
  catch (const ptree_error &e) 
  {
    LOG(WARNING) << "When paring " << annoname << ": " << e.what();
  }
  
  datum->clear_float_data();
  BOOST_FOREACH(ptree::value_type &v1, pt.get_child("annotation")) 
  {
    if (v1.first == "object") 
	{
      ptree object = v1.second;
      int label(-1);
      vector<float> box(4, 0);
      int difficult(0);
      BOOST_FOREACH(ptree::value_type &v2, object.get_child("")) 
	  {
        ptree pt2 = v2.second;
        if (v2.first == "name") 
		{
          string name = pt2.data();
          //LOG(INFO) << " NAME:  " << name;
          // map name to label
		  // 标注文件 name标签 : 实际类别idclass_id
          label = name_to_label(name, label_map);
          //label = pt2.data();
          if (label < 0) 
		  {
            LOG(FATAL) << "Anno file " << annoname << " -> unknown name: " << name;
          }
        } 
		else if (v2.first == "bndbox") 
		{
          int xmin = pt2.get("xmin", 0);//左下角  坐标
          int ymin = pt2.get("ymin", 0);
          int xmax = pt2.get("xmax", 0);//右上角  坐标
          int ymax = pt2.get("ymax", 0);
		  // 范围限制
          LOG_IF(WARNING, xmin < 0 || xmin > ori_w) << annoname << 
              " bounding box exceeds image boundary";
          LOG_IF(WARNING, xmax < 0 || xmax > ori_w) << annoname << 
              " bounding box exceeds image boundary";
          LOG_IF(WARNING, ymin < 0 || ymin > ori_h) << annoname << 
              " bounding box exceeds image boundary";
          LOG_IF(WARNING, ymax < 0 || ymax > ori_h) << annoname << 
              " bounding box exceeds image boundary";
          LOG_IF(WARNING, xmin > xmax) << annoname << 
              " bounding box exceeds image boundary";
          LOG_IF(WARNING, ymin > ymax) << annoname << 
              " bounding box exceeds image boundary";
		  // 计算 边框中心坐标 归一化到原图像的0~1
          box[0] = float(xmin + (xmax - xmin) / 2.) / ori_w;
          box[1] = float(ymin + (ymax - ymin) / 2.) / ori_h;
		  // 计算边框尺寸, 归一化到 到原图像的0~1
          box[2] = float(xmax - xmin) / ori_w;
          box[3] = float(ymax - ymin) / ori_h;
        } 
		else if (v2.first == "difficult") 
		{
          difficult = atoi(pt2.data().c_str());
        }
      }
      CHECK_GE(label, 0) << "label must start at 0";
      datum->add_float_data(float(label));	    //label  标签
      datum->add_float_data(float(difficult));	//diff   偏移?
      for (int i = 0; i < 4; ++i) 
	  {
        datum->add_float_data(box[i]);	//box[4]  x,y,w,h  (范围0~1)
      }
    }
  }
}
///////////////////////////////////////

// 文件转成 datum  分类数据=========================
bool ReadFileToDatum(
const string& filename, 
const int label,
Datum* datum) 
{
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);// 图片数据
    datum->set_label(label);// 类别id
    datum->set_encoded(true);// 编码标志
    return true;
  } else {
    return false;
  }
}

/////////////////////////////////add/////////////
// 文件转成 datum  检测数据=========================
bool ReadFileToDatum(
const string& filename, // 图片 
const string& annoname, // xml 标注文件
const map<string, int>& label_map, // 标注文件 name标签 : class_id
int ori_w, // 原始图像尺寸
int ori_h, 
Datum* datum) 
{
  
  std::streampos size;
  // 文件格式读取 图片
  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) 
  {
    size = file.tellg();//大小
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
	// 设置数据 datum
    datum->set_data(buffer);
    datum->set_encoded(true);
    ParseXmlToDatum(annoname, label_map, ori_w, ori_h, datum);
    return true;
  } else {
    return false;
  }
}
////////////////////////////////////////////

bool MapLabelToName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_name) {
  // cleanup
  label_to_name->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& name = map.item(i).name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!label_to_name->insert(std::make_pair(label, name)).second) {
        LOG(FATAL) << "There are many duplicates of label: " << label;
        return false;
      }
    } else {
      (*label_to_name)[label] = name;
    }
  }
  return true;
}

bool MapLabelToDisplayName(const LabelMap& map, const bool strict_check,
    std::map<int, string>* label_to_display_name) {
  // cleanup
  label_to_display_name->clear();

  for (int i = 0; i < map.item_size(); ++i) {
    const string& display_name = map.item(i).display_name();
    const int label = map.item(i).label();
    if (strict_check) {
      if (!label_to_display_name->insert(
              std::make_pair(label, display_name)).second) {
        LOG(FATAL) << "There are many duplicates of label: " << label;
        return false;
      }
    } else {
      (*label_to_display_name)[label] = display_name;
    }
  }
  return true;
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void EncodeCVMatToDatum(const cv::Mat& cv_img, const string& encoding,
                        Datum* datum) {
  std::vector<uchar> buf;
  cv::imencode("."+encoding, cv_img, buf);
  datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                              buf.size()));
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_encoded(true);
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  // 图片数据存储格式为0~255 8位
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());// 1/3 图像通道数量
  datum->set_height(cv_img.rows);// 行
  datum->set_width(cv_img.cols); // 列
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);// 未编码
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  // 总大小 字节大小
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  // 设置数据
  datum->set_data(buffer);
}
#endif  // USE_OPENCV
}  // namespace caffe
