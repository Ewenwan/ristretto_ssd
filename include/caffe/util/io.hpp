#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace caffe {

using ::google::protobuf::Message;
using ::boost::filesystem::path;

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  const path& model =
    boost::filesystem::temp_directory_path()/"caffe_test.%%%%-%%%%";
  for ( int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++ ) {
    const path& dir = boost::filesystem::unique_path(model).string();
    bool done = boost::filesystem::create_directory(dir);
    if ( done ) {
      *temp_dirname = dir.string();
      return;
    }
  }
  LOG(FATAL) << "Failed to create a temporary directory.";
}

inline void MakeTempFilename(string* temp_filename) {
  static path temp_files_subpath;
  static uint64_t next_temp_file = 0;
  temp_filename->clear();
  if ( temp_files_subpath.empty() ) {
    string path_string="";
    MakeTempDir(&path_string);
    temp_files_subpath = path_string;
  }
  *temp_filename =
    (temp_files_subpath/caffe::format_int(next_temp_file++, 9)).string();
}

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool MapLabelToName(const LabelMap& map, const bool strict_check,
                    std::map<int, string>* label_to_name);

inline bool MapLabelToName(const LabelMap& map,
                           std::map<int, string>* label_to_name) {
  return MapLabelToName(map, true, label_to_name);
}
bool MapLabelToDisplayName(const LabelMap& map, const bool strict_check,
                           std::map<int, string>* label_to_display_name);

inline bool MapLabelToDisplayName(const LabelMap& map,
                              std::map<int, string>* label_to_display_name) {
  return MapLabelToDisplayName(map, true, label_to_display_name);
}

bool DecodeDatumNative(Datum* datum);
bool DecodeDatum(Datum* datum, bool is_color);

#ifdef USE_OPENCV

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

	
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

void EncodeCVMatToDatum(const cv::Mat& cv_img, const string& encoding,
                        Datum* datum);

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);

///////////////检测数据 需要返回原图像大小//////
cv::Mat ReadImageToCVMat(
    const string& filename,// 图片
    const int height,      // 目标尺寸
	const int width, 
	const bool is_color,   // 彩色?
    int* ori_w,            // 原图像尺寸
	int* ori_h);
////////////////////////////////////

//////////////////////add//////
// 检测数据转换 
// 图片+ 标注文件  转换成 数据  Datum* datum
bool ReadBoxDataToDatum(
	const string& filename, // 图片路径 
	const string& annoname, // xml标注文件路径
	const map<string, int>& label_map, // 标注文件 name标签 : class_id
	const int height,       // 尺寸变形 固定到网络输入大小
	const int width, 
	const bool is_color,    // 彩色图像
	const std::string & encoding, // 编码格式 
	Datum* datum);
///////////////////////////////

#endif  // USE_OPENCV

// xml 格式数据转换成 datum  检测数据===========================
//  这里要区分是 xml/json/txt格式数据==========待修改
void ParseXmlToDatum(
const string& annoname, // xml 标注文件 
const map<string, int>& label_map,// 标注文件 name标签 : class_id
int ori_w, // 元图片尺寸
int ori_h, 
Datum* datum); 

////////////////////////
// 文件转成 datum  检测数据=========================
bool ReadFileToDatum(
const string& filename, // 图片 
const string& annoname, // xml 标注文件
const map<string, int>& label_map, // 标注文件 name标签 : class_id
int ori_w, // 原始图像尺寸
int ori_h, 
Datum* datum);
///////////////////////////////////


}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
