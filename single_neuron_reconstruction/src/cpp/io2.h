#ifndef _IO2_
#define _IO2_

#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <vector>
#include <QtGui>
#include <QtCore>
#include <QDateTime>
#include <QApplication>
#include <unordered_set>
#include <unordered_map>

#include "v3d_interface.h"
#include "basic_surf_objs.h"
#include "data_definition.h"

using namespace std;

V3DLONG* getDimTeraFly1(V3DPluginCallback2& callback,QString &input);
bool getLandmarkTeraFly1(V3DPluginCallback2& callback,V3DPluginArgList &input, V3DPluginArgList &output);
bool getSubVolumeFromTeraFly1(V3DPluginCallback2 &callback, char *imagePath, Image4DSimple &subVolumeImage, 
                            V3DLONG xb, V3DLONG xe, V3DLONG yb, V3DLONG ye, V3DLONG zb, V3DLONG ze,V3DLONG *originSize);

bool readSWCtoNodeList(const string filePath, NodeList &nt);
bool saveMarker_file1(const char* marker_file, vector<MyMarker *> &outmarkers);    
bool saveSWCFile(const string savefile, NodeList& neuronTree, bool verbose= false);

bool normalization(Image4DSimple *image);
void file_copy(string src, string target);

std::vector<std::pair<std::string, std::string>> LoadConfig(const std::string& yaml_path);
std::string GetCfg(const std::vector<std::pair<std::string,std::string>>& cfgs,const std::string& key);
#endif //_IO2_