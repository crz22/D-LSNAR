#include "io2.h"

//===================================================================
// getLandmarkTeraFly1
//===================================================================
bool getLandmarkTeraFly1(V3DPluginCallback2& callback,V3DPluginArgList &input, V3DPluginArgList &output){

    if (input.empty() || output.empty()) return false;

    auto* inVec  = static_cast<std::vector<char*>*>(input.at(0).p);
    auto* outVec = static_cast<std::vector<char*>*>(output.at(0).p);

    if (!inVec || !outVec || inVec->empty() || outVec->empty()) return false;

    // get landmark
    std::vector<MyMarker*> markers;
    for (const auto& m : callback.getLandmarkTeraFly())
        markers.push_back(new MyMarker(m.x, m.y, m.z));

    if (markers.empty()) return false;

    bool ok = saveMarker_file1((*outVec)[0], markers);

    // Release MyMarker memory
    for (auto* p : markers) delete p;
    markers.clear();

    return ok;
}

//===================================================================
// saveMarker_file1
//===================================================================
bool saveMarker_file1(const char* marker_file, vector<MyMarker *> &outmarkers) {

    if (!marker_file) return false;

    cout << "[Info] Saving " << outmarkers.size() << " markers to " << marker_file << std::endl;

    std::ofstream ofs(marker_file, std::ios::app);
    if (!ofs) {
        std::cerr << "[ERROR] Cannot open marker file: " << marker_file << std::endl;
        return false;
    }

    for (const auto* m : outmarkers) {
        ofs << m->x << "," << m->y << "," << m->z << ","
            << 0 << "," << 0 << ","           // radius, shape
            << "" << "," << "" << ","          // name, comment
            << 255 << "," << 0 << "," << 0    // color R,G,B
            << "\n";
    }
    return true;
}

//===================================================================
// saveSWCFile
//===================================================================
bool saveSWCFile(const string savefile, NodeList& neuronTree, bool verbose)
{
    // if (neuronTree.isEmpty()) {
    //     if (verbose)
    //         std::cout << "[WARN] Empty neuron tree, skip saving: " << savefile << std::endl;
    //     return false;
    // }

    // Delete old files
    QFile file(QString::fromStdString(savefile));
    if (file.exists()) file.remove();

    std::ofstream ofs(savefile, std::ios::binary);
    if (!ofs) {
        if (verbose)
            std::cerr << "[ERROR] Failed to save: " << savefile << std::endl;
        return false;
    }

     // Establish node index
    V3DLONG nums = 0;
    std::unordered_map<NeuronNode*, V3DLONG> ind;
    for (NeuronNode* node : neuronTree)
        ind[node] = ++nums;

    ofs << "# name "    << savefile << "\n"
        << "# comment " << "\n"
        << "# n, type, x, y, z, radius, parent\n";

    nums = 0;
    for (NeuronNode* node : neuronTree) {
        const V3DLONG parent_id =
            (node->parent && ind.count(node->parent)) ? ind[node->parent] : -1;

        ofs << ++nums << " " << node->type << " "
            << std::fixed << std::setprecision(3)
            << node->x << " " << node->y << " "
            << node->z << " " << node->radius << " "
            << parent_id << "\n";
    }

    if (verbose)
        std::cout << "[Info] Saved " << nums << " nodes to " << savefile << std::endl;

    return true;
}

//===================================================================
// getDimTeraFly1
//===================================================================
V3DLONG* getDimTeraFly1(V3DPluginCallback2& callback,QString &input){
    
    V3DLONG* imageSize = new V3DLONG[4]();

    callback.getDimTeraFly(input.toStdString(), imageSize);  

    cout <<"[Info] Image size: "<< imageSize[0] << " " << imageSize[1] << " " << imageSize[2] << " " << imageSize[3] <<endl;

    return imageSize;
}

//===================================================================
// getSubVolumeFromTeraFly1
//===================================================================
bool getSubVolumeFromTeraFly1(V3DPluginCallback2 &callback, char *imagePath, Image4DSimple &subVolumeImage, V3DLONG xb,
                             V3DLONG xe, V3DLONG yb, V3DLONG ye, V3DLONG zb, V3DLONG ze,V3DLONG *originSize) {

    if (!imagePath || !originSize) return false;
    
    const V3DLONG xdim = std::min(xe, originSize[0]) - xb;
    const V3DLONG ydim = std::min(ye, originSize[1]) - yb;
    const V3DLONG zdim = std::min(ze, originSize[2]) - zb;
    const V3DLONG cdim = originSize[3];

    if (xdim <= 0 || ydim <= 0 || zdim <= 0) {
        std::cerr << "[ERROR] Invalid sub-volume dimensions: "
                  << xdim << " x " << ydim << " x " << zdim << std::endl;
        return false;
    }

    unsigned char* data = callback.getSubVolumeTeraFly(imagePath,
                                                        xb, xb + xdim,
                                                        yb, yb + ydim,
                                                        zb, zb + zdim);

    if (!data) {
        std::cerr << "[ERROR] getSubVolumeTeraFly returned null." << std::endl;
        return false;
    }

    subVolumeImage.setData(data, xdim, ydim, zdim, cdim, V3D_UINT8);
    subVolumeImage.setOriginX(static_cast<double>(xb));
    subVolumeImage.setOriginY(static_cast<double>(yb));
    subVolumeImage.setOriginZ(static_cast<double>(zb));

    return true;
}

//===================================================================
// normalization
//===================================================================
bool normalization(Image4DSimple *image) {
    if (!image) return false;

    const V3DLONG totalSize = image->getTotalUnitNumber();
    auto* raw = image->getRawData();
    if (!raw || totalSize == 0) return false;

    unsigned char maxVal = 0, minVal = 255;
    for (V3DLONG i = 0; i < totalSize; ++i) {
        maxVal = std::max(raw[i], maxVal);
        minVal = std::min(raw[i], minVal);
    }

    if (maxVal == minVal) {
        std::cerr << "[WARN] max == min, cannot normalize." << std::endl;
        return false;
    }

    const double scale = 255.0 / (maxVal - minVal);
    for (V3DLONG i = 0; i < totalSize; ++i) {
        raw[i] = static_cast<unsigned char>(
            std::round((raw[i] - minVal) * scale));
    }
    return true;
}

//===================================================================
// readSWCtoNodeList
//===================================================================
bool readSWCtoNodeList(const std::string& filePath, NodeList& nt)
{
    std::ifstream ifs(filePath);
    if (!ifs) {
        std::cerr << "[ERROR] Open failed: " << filePath << std::endl;
        return false;
    }

    nt.clear();
    std::unordered_map<int, NeuronNode*> marker_map;
    std::unordered_map<NeuronNode*, int> parid_map;

    for (std::string line; std::getline(ifs, line); ) {
        // 跳过注释行和空行
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        int id = -1, par_id = -1;

        NeuronNode* pNode = new NeuronNode;
        ss >> id >> pNode->type
           >> pNode->x >> pNode->y >> pNode->z
           >> pNode->radius >> par_id;

        if (ss.fail() || id == -1) {
            delete pNode;
            continue;
        }

        marker_map[id]   = pNode;
        parid_map[pNode] = par_id;
        nt.addNode(pNode);
    }

    // 重建父子关系
    for (NeuronNode* pNode : nt) {
        const int parid = parid_map[pNode];
        if (parid == -1) continue;
        auto it = marker_map.find(parid);
        if (it == marker_map.end()) continue;
        pNode->parent = it->second;
        pNode->parent->children.push_back(pNode);
    }

    std::cout << "[Info] Read " << nt.size()
              << " nodes from " << filePath << std::endl;
    return true;
}


//===================================================================
// file_copy
//===================================================================
void file_copy(const std::string& src, const std::string& target)
{
    const QString srcPath    = QString::fromStdString(src);
    const QString targetPath = QString::fromStdString(target);

    QFile file(srcPath);
    if (file.copy(targetPath))
        std::cout << "[Info] File copied: " << src << " -> " << target << std::endl;
    else
        std::cerr << "[ERROR] Copy failed: " << file.errorString().toStdString() << std::endl;
}


//===================================================================
// LoadConfig
//===================================================================
std::vector<std::pair<std::string, std::string>> LoadConfig(const std::string& yaml_path)
{
    std::vector<std::pair<std::string, std::string>> config;

    std::ifstream fin(yaml_path.c_str());

    if (!fin.is_open())
    {
        std::cerr << "Cannot open config file: "
                  << yaml_path << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(fin, line))
    {
        // 去除注释
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos)
        {
            line = line.substr(0, comment_pos);
        }

        // 跳过空行
        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;

        size_t pos = line.find(':');

        if (pos == std::string::npos)
            continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        // 去除前后空格
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);

        value.erase(0, value.find_first_not_of(" \t"));

        size_t end = value.find_last_not_of(" \t\r\n");
        if (end == std::string::npos)
            value.clear();
        else
            value.erase(end + 1);

        config.emplace_back(key, value);
    }

    fin.close();

    return config;
}

//===================================================================
// GetCfg
//===================================================================
std::string GetCfg(
    const std::vector<std::pair<std::string,std::string>>& cfgs,
    const std::string& key)
{
    for (const auto& kv : cfgs)
    {
        if (kv.first == key)
            return kv.second;
    }
    
    std::cerr << "[WARN] Config key not found: " << key << std::endl;
    return "";
}