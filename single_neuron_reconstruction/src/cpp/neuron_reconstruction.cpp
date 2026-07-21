#include "neuron_reconstruction.h"

Neuron_Reconstruction :: Neuron_Reconstruction(V3DPluginCallback2& cb):
callback(cb)
{   
    cout<<"Neuron_Reconstruction start"<<endl;
}

//===================================================================
// Initialize the output directory
//===================================================================
bool Neuron_Reconstruction::initOutputDirectories()
{
    if (outputFolderPath.isEmpty())
        outputFolderPath = "temp";

    QDateTime dateTime = QDateTime::currentDateTime();
    outputFolderPath += "/" + dateTime.toString("yyyy-MM-dd/hh_mm_ss");

    QDir outputDir(outputFolderPath);
    if (!outputDir.exists() && !outputDir.mkpath(".")) {
        std::cerr << "[ERROR] Cannot create output dir: "
                  << outputFolderPath.toStdString() << std::endl;
        return false;
    }

    QDir imageDir(imagePath);
    imageDir.cdUp();
    const QString tmpDirName = imageDir.dirName() + "_tmp";
    outputDir.mkpath(tmpDirName);
    tempFolderPath = outputFolderPath + "/" + tmpDirName;

    finalSWCfile = QString("%1/%2_%3.swc")
                       .arg(outputFolderPath, imageDir.dirName(), traceMethod);

    //finalSWCfile = QString("%1/%2_%3").arg(outputFolderPath, imageDir.dirName(), traceMethod);
    //finalSWCfile.append(".swc");
    return true;
}

//===================================================================
// Save Configuration
//===================================================================
bool Neuron_Reconstruction :: saveConfiguration(const QString &savePath) {
    QFile outFile(savePath);
    if (outFile.exists()) outFile.remove();
    if (!outFile.open(QFile::ReadWrite)) return false;

    QTextStream textStream(&outFile);
    const auto& cfgs = Cfgs;

    if (!cfgs.empty())
    {
        for (const auto& kv : cfgs)
        {
            textStream<< QString::fromStdString(kv.first)<< ": "<< QString::fromStdString(kv.second)<< endl;
        }
    }
    else
    {
        textStream << "traceMethod: "<< traceMethod<<endl;
        textStream << "segmentMethod: "<< segmentMethod<<endl;
        textStream << "blockSize: "<< blockSize<<endl;
        textStream << "node_step: "<< node_step<<endl; 
        textStream << "branch_MAXL: "<< branch_MAXL<<endl;
        textStream << "Angle_T: " << Angle_T << endl;
        textStream << "Lamd: " << Lamd<< endl;
    }
    
    //textStream << "OutputPath: " << QFileInfo(savePath).absolutePath() << endl;
    //textStream << "OutputPath: " << QFileInfo(savePath).absolutePath() << endl;
    outFile.close();
    return true;
}

//===================================================================
// Initialize tracking parameters
//===================================================================
bool Neuron_Reconstruction::initTracingArgs(V3DPluginArgList& tracingArgsList)
{
    tracingArgsList.clear();
    tracingArgsList.push_back(V3DPluginArgItem());
    tracingArgsList.push_back(V3DPluginArgItem());

    parasString.clear();
    parasString.reserve(3);
    parasString.emplace_back(pytorchPath.toStdString());
    parasString.emplace_back(configurationPath.toStdString());
    parasString.emplace_back(tempFolderPath.toStdString());

    paras.clear();
    paras.reserve(parasString.size());
    for (auto& s : parasString)
        paras.push_back(const_cast<char*>(s.c_str()));

    subImageVec.assign(1, nullptr);

    tracingArgsList.front().p = &subImageVec;
    tracingArgsList.back().p  = &paras;
    return true;
}

//===================================================================
// TeraFly initialization and load marker
//===================================================================
//Select Marker from Whole Brain Images
bool Neuron_Reconstruction::fetchAndReadTeraFlyMarker(const string& markerFilePath)
{
    V3DPluginArgList inputArgsList, outputArgsList;
    vector<char*> unused{nullptr};
    // 注意：markerFilePath 必须在整个调用期间有效
    vector<char*> output{const_cast<char*>(markerFilePath.data())};

    inputArgsList.push_back(V3DPluginArgItem());
    inputArgsList.front().p = &unused;
    outputArgsList.push_back(V3DPluginArgItem());
    outputArgsList.front().p = &output;

    getLandmarkTeraFly1(callback, inputArgsList, outputArgsList);

    if (!readStartMarkers(markerFilePath.data()) || allTargetList.empty()) {
        std::cerr << "[ERROR] No marker found in whole brain image." << std::endl;
        return false;
    }
    return true;
}

bool Neuron_Reconstruction::initTeraFly()
{
    if (!terafly) {
        std::cerr << "[ERROR] No terafly mode supported!" << std::endl;
        return false;
    }

    // load marker file
    if (markerPath.isEmpty()) {
        const string markerFilePath = (outputFolderPath + "/start.marker").toStdString();
        if (!fetchAndReadTeraFlyMarker(markerFilePath)) return false;
    } 
    else {
        if (!readStartMarkers(markerPath)) {
            std::cerr << "[ERROR] Failed to read marker: " << markerPath.toStdString() << std::endl;
            return false;
        }
        if (allTargetList.empty()) {
            std::cerr << "[ERROR] Empty marker file: " << markerPath.toStdString() << std::endl;
            return false;
        }
    }

    // Leave only one starting point
    while (allTargetList.size() > 1) {
        delete allTargetList.front();
        allTargetList.pop_front();
    }

    // get size of big image
    imageSize = getDimTeraFly1(callback, imagePath);
    return true;
}

//===================================================================
// Initialize the starting block
//===================================================================
bool Neuron_Reconstruction::initStartBlocks()
{
    if (allTargetList.empty()) {
        std::cerr << "[ERROR] No start block." << std::endl;
        return false;
    }
    // Only one starting point is allowed
    BlockSimple* pStart = allTargetList.front();
    pStart->originX    -= blockSize / 2;
    pStart->originY    -= blockSize / 2;
    pStart->originZ    -= blockSize / 2;
    pStart->isStartBlock = true;
    return true;
}

//===================================================================
// Main reconstruction Loop
//===================================================================
void Neuron_Reconstruction :: Block_boundary_adjust(BlockSimple *pBlock){
    pBlock->originX = (pBlock->originX > 0) ? pBlock->originX : 0;
    pBlock->originY = (pBlock->originY > 0) ? pBlock->originY : 0;
    pBlock->originZ = (pBlock->originZ > 0) ? pBlock->originZ : 0;
    pBlock->blockSizeX = (pBlock->originX + blockSize) < imageSize[0] ? blockSize : imageSize[0] - pBlock->originX;
    pBlock->blockSizeY = (pBlock->originY + blockSize) < imageSize[1] ? blockSize : imageSize[1] - pBlock->originY;
    pBlock->blockSizeZ = (pBlock->originZ + blockSize) < imageSize[2] ? blockSize : imageSize[2] - pBlock->originZ;
}

// --------------------------------------------------------- //
//Boundary node search
void Neuron_Reconstruction :: SearchNodesOnBoundary(BlockSimple *Target, NodeList &nt, int margin){
    const auto& os  = Target->originX;
    const auto& oy  = Target->originY;
    const auto& oz  = Target->originZ;
    const auto& bsx = Target->blockSizeX;
    const auto& bsy = Target->blockSizeY;
    const auto& bsz = Target->blockSizeZ;

    if (os  != 0)            findTips(nt, 0, margin,       Direction::LeftSide);
    if (os  + bsx != imageSize[0]) findTips(nt, bsx - margin, bsx, Direction::RightSide);
    if (oy  != 0)            findTips(nt, 0, margin,       Direction::UpSide);
    if (oy  + bsy != imageSize[1]) findTips(nt, bsy - margin, bsy, Direction::DownSide);
    if (oz  != 0)            findTips(nt, 0, margin,       Direction::OutSide);
    if (oz  + bsz != imageSize[2]) findTips(nt, bsz - margin, bsz, Direction::InSide);
}

//Insert nodes at the boundary
void Neuron_Reconstruction::findTips(
    NodeList& nt, V3DLONG start, V3DLONG end, Direction direction)
{
    // Lambda: Determine whether the node is within the boundary area (overlapping area of two image blocks)
    auto inBound = [&](NeuronNode* m) -> bool {
        switch (direction) {
            case Direction::LeftSide:  return m->x < end;
            case Direction::RightSide: return m->x > start;
            case Direction::UpSide:    return m->y < end;
            case Direction::DownSide:  return m->y > start;
            case Direction::OutSide:   return m->z < end;
            case Direction::InSide:    return m->z > start;
        }
        return false;
    };

    const double boundary = (direction == Direction::LeftSide  || direction == Direction::UpSide  || direction == Direction::OutSide)
                                ? static_cast<double>(end) : static_cast<double>(start);

    NodeList tipList;
    for (NeuronNode* m : nt)
        if (inBound(m)) tipList.push_back(m);

    //Insert point in the boundary
    for (NeuronNode* tip : tipList) {
        // Child nodes outside the boundary → interpolation
        for (NeuronNode* child : tip->children) {
            if (tipList.indexOf(child) != -1) continue;
            nt.addNode(interpolateNodeOnBoundary(child, boundary, direction));
        }
        // Parent node outside the boundary → Interpolation
        if (tip->parent && tipList.indexOf(tip->parent) == -1)
            nt.addNode(interpolateNodeOnBoundary(tip, boundary, direction));
    }

    //clear tipList
    while (!tipList.empty()) {
        NeuronNode* node = tipList.front();
        tipList.pop_front();
        nt.removeOne(node); //** */
        delete node;
    }
}

// --------------------------------------------------------- //
//Cut off small branches
bool Neuron_Reconstruction::pruneTinyBranch(
    int length, NodeList* nodeList, BlockSimple* blockSimple)
{
    if (!nodeList || !blockSimple) {
        std::cout << "pruneTinyBranch: invalid arguments." << std::endl;
        return false;
    }

    NodeList tinyBack, tinyFront;

    for (NeuronNode* node : *nodeList) {
        if (!node->children.empty()) continue;

        // Skip nodes near the margin
        const double x = node->x, y = node->y, z = node->z;
        const double bsx = blockSimple->blockSizeX;
        const double bsy = blockSimple->blockSizeY;
        const double bsz = blockSimple->blockSizeZ;

        if (std::abs(x - marginSize) < 3 || std::abs(x - bsx + marginSize) < 3) continue;
        if (std::abs(y - marginSize) < 3 || std::abs(y - bsy + marginSize) < 3) continue;
        if (std::abs(z - marginSize) < 3 || std::abs(z - bsz + marginSize) < 3) continue;

        //
        NeuronNode* bp    = node;
        int         count = 0;
        while (bp->parent && bp->parent->children.size() < 2 && count < length) {
            bp = bp->parent;
            ++count;
        }
        if (count < length && bp->radius < 20) {
            tinyBack.push_back(node);
            tinyFront.push_back(bp);
        }
    }

    // Disconnect the branch root node from its parent node
    for (NeuronNode* front : tinyFront) {
        if (!front->parent) continue;
        front->parent->children.removeOne(front);
        front->parent = nullptr;
    }

    // Remove from leaf node upwards
    for (NeuronNode* back : tinyBack) {
        NeuronNode* cur = back;
        while (cur) {
            NeuronNode* next = cur->parent;
            delete cur;
            cur = next;
        }
    }

    std::cout << "[Info] Pruned " << tinyFront.size() << " branch(es)." << std::endl;
    return !tinyFront.empty();
}

// --------------------------------------------------------- //
//Block neuron reconstruction
bool Neuron_Reconstruction :: Block_Neuron_Reconstruct(
    BlockSimple *currentTarget,
    V3DPluginArgList &args, 
    NodeList &blockNeuronTree, 
    const string& coordinateString,
    V3DLONG OriginX, V3DLONG OriginY, V3DLONG OriginZ )
{
    cout << "[Info] Tracing..." << endl;

    //--- Extract sub images ---
    //Image blocks are named with the coordinates of the upper left point

    currentSubImagePath = QString("%1/x%2_y%3_z%4.tif") .arg(tempFolderPath) .arg(OriginX).arg(OriginY).arg(OriginZ).toStdString();

    Image4DSimple subImage = Image4DSimple();

    if (terafly) {
        const string imgPathStd = imagePath.toStdString();
        if (!getSubVolumeFromTeraFly1(
                callback,
                const_cast<char*>(imgPathStd.data()),
                subImage,
                OriginX, OriginX + currentTarget->blockSizeX,
                OriginY, OriginY + currentTarget->blockSizeY,
                OriginZ, OriginZ + currentTarget->blockSizeZ,
                imageSize))
        {
            std::cerr << "[ERROR] Failed to get subvolume from TeraFly." << std::endl;
            return false;
        }
    } 
    else {
        std::cerr << "[ERROR] Only TeraFly image is supported." << std::endl;
        return false;
    }

    if (!callback.saveImage(&subImage, const_cast<char*>(currentSubImagePath.c_str()))) {
        std::cerr << "[ERROR] Failed to save sub image: " << currentSubImagePath << std::endl;
        return false;
    }

    // --- Neuron reconstruction ---
    if (!D_LSNARS(callback, args, currentSubImagePath, pythonCodePath, currentTarget->isStartBlock)) {
        std::cerr << "[ERROR] D_LSNARS failed." << std::endl;
        return false;
    }

    // --- Read resampling results ---
    if (!readBlockSWC(QString::fromStdString(coordinateString), ".tif_resample.swc", blockNeuronTree))
        return false;

    // --- pruning ---
    //Delete branches with fewer than min_branch_length
    const std::string removeTipsFile = currentSubImagePath + "_resample.swc_removeTips.swc";
    SearchNodesOnBoundary(currentTarget, blockNeuronTree, marginSize);
    while (pruneTinyBranch(min_branch_length, &blockNeuronTree, currentTarget)); // Repeatedly cut until there are no more branches to cut
    saveSWCFile(removeTipsFile, blockNeuronTree);

    // ---  local coordinates → global coordinates ---
    for (NeuronNode* m : blockNeuronTree) {
        m->x += static_cast<double>(OriginX);
        m->y += static_cast<double>(OriginY);
        m->z += static_cast<double>(OriginZ);
    }

    const std::string notConnectFile = currentSubImagePath + "_not_connect.swc";
    saveSWCFile(notConnectFile, blockNeuronTree);
    return true;
}

// --------------------------------------------------------- //
void Neuron_Reconstruction::extractSomaNodes(NodeList& src, NodeList& dst)
{
    for (NeuronNode* node : src) {
        if (node->type == somatype)
            dst.addNode(node);
    }
}

void Neuron_Reconstruction::collectLeafNodes(const NodeList& src, NodeList& dst)
{
    for (NeuronNode* node : src) {
        if (node->children.empty())
            dst.push_back(node);
    }
}

// --------------------------------------------------------- //
// Search for 26 neighborhood reconstructed blocks
void Neuron_Reconstruction::SearchNearBlock(BlockSimple* centralblock, BlockSimpleList& candidateNeighbours)
{
    const V3DLONG ox  = centralblock->originX;
    const V3DLONG oy  = centralblock->originY;
    const V3DLONG oz  = centralblock->originZ;
    const V3DLONG bsx = centralblock->blockSizeX;
    const V3DLONG bsy = centralblock->blockSizeY;
    const V3DLONG bsz = centralblock->blockSizeZ;

    for (int zz = -1; zz <= 1; ++zz) {
        if (zz == -1 && oz == 0) continue;
        if (zz ==  1 && oz + bsz == imageSize[2]) continue;
        const V3DLONG zBS = (zz == 1) ? bsz : blockSize;

        for (int yy = -1; yy <= 1; ++yy) {
            if (yy == -1 && oy == 0) continue;
            if (yy ==  1 && oy + bsy == imageSize[1]) continue;
            const V3DLONG yBS = (yy == 1) ? bsy : blockSize;

            for (int xx = -1; xx <= 1; ++xx) {
                if (xx == -1 && ox == 0) continue;
                if (xx ==  1 && ox + bsx == imageSize[0]) continue;
                if (xx == 0 && yy == 0 && zz == 0) continue;
                const V3DLONG xBS = (xx == 1) ? bsx : blockSize;

                V3DLONG nx = ox + xx * (xBS - 2 * marginSize);
                V3DLONG ny = oy + yy * (yBS - 2 * marginSize);
                V3DLONG nz = oz + zz * (zBS - 2 * marginSize);
                nx = std::max<V3DLONG>(nx, 0);
                ny = std::max<V3DLONG>(ny, 0);
                nz = std::max<V3DLONG>(nz, 0);

                const string key =
                    QString("x%1_y%2_z%3").arg(nx).arg(ny).arg(nz).toStdString();

                auto it = blockMap.find(key);
                if (it != blockMap.end())
                    candidateNeighbours.push_back(it->second);
            }
        }
    }
}

// Extract neuronal branches (BFS starting from the root node)
bool Neuron_Reconstruction::extractNeuronSegment(NodeList& nt, NeuronNode* pNode, NodeList& neuronSeg)
{
    if (nt.indexOf(pNode) == -1) return false;

    // Find root node
    while (pNode->parent) pNode = pNode->parent;

    neuronSeg.clear();
    NodeList queue;
    queue.push_back(pNode);

    while (!queue.empty()) {
        NeuronNode* node = queue.front();
        neuronSeg.addNode(node);
        for (NeuronNode* child : node->children) queue.push_back(child);
        queue.pop_front();
    }
    return true;
}

// Reset the root node
bool Neuron_Reconstruction::setNewRoot(NeuronNode* newRoot)
{
    if (!newRoot) return false;

    NeuronNode* prev = nullptr;
    NeuronNode* cur  = newRoot;
    NeuronNode* next = cur->parent;

    while (cur) {
        if (cur->parent)
            cur->parent->children.removeOne(cur);

        cur->parent = prev;

        if (cur->parent)
            cur->parent->children.push_back(cur);

        prev = cur;
        cur  = next;

        next = (next) ? next->parent : nullptr;
    }
    return true;
}

// Connect two nodes
void Neuron_Reconstruction::linkNodes(NeuronNode* pNode, NeuronNode* pConnect, bool useEdge){
    if (!useEdge) {
        // Merge: Transfer the child nodes of pNode to pConnect, and then delete pNode
        while (!pNode->children.empty()) {
            NeuronNode* child   = pNode->children.front();
            child->parent       = pConnect;
            pConnect->children.push_back(child);
            pNode->children.pop_front();
        }
        delete pNode;
    } 
    else {
        // connect
        pNode->parent = pConnect;
        pConnect->children.push_back(pNode);
    }
}

// Determine whether searchNode is in the branch starting from branchNode
bool Neuron_Reconstruction::findNodeInBranch(NeuronNode* branchNode, NeuronNode* searchNode){
    if (!branchNode || !searchNode) return false;

    while (branchNode->parent) branchNode = branchNode->parent;

    NodeList queue;
    queue.push_back(branchNode);
    while (!queue.empty()) {
        NeuronNode* node = queue.front();
        queue.pop_front();
        if (node == searchNode) return true;
        for (NeuronNode* child : node->children)
            queue.push_back(child);
    }
    return false;
}

//Deal with other connectable points on the branch
void Neuron_Reconstruction::connectBranchNodes(
    NodeList& branch,
    std::unordered_map<NeuronNode*, NeuronNode*>& connectPoint,
    std::unordered_map<NeuronNode*, double>&       connectDist,
    double fusionDist)
{
    // Collect the nodes to be connected on this branch
    NodeList pendingList;
    for (NeuronNode* n : branch) {
        if (connectPoint.count(n))
            pendingList.push_back(n);
    }

    while (!pendingList.empty()) {

        //Find the point with the smallest connection distance
        NeuronNode* pNode   = nullptr;
        double      minDist = std::numeric_limits<double>::max();
        for (NeuronNode* c : pendingList) {
            if (connectDist[c] < minDist) {
                minDist = connectDist[c];
                pNode   = c;
            }
        }
        NeuronNode* pConnect = connectPoint[pNode];
        connectPoint.erase(pNode);
        connectDist.erase(pNode);
        pendingList.removeOne(pNode);

        // prevent loop formation
        if (findNodeInBranch(pNode, pConnect)) continue;

        if (pNode->parent && !setNewRoot(pNode)) continue;
        linkNodes(pNode, pConnect, fusionDist < minDist);
    }
}

// Connect the current block with surrounding blocks
bool Neuron_Reconstruction::connect(
    BlockSimpleList& candidateGroups,
    NodeList& neuronTree,
    NodeList& ConnectedSegs,
    double thresDist)
{
    const double fusionDist = 1.0;

    // Find the nearest candidate connection point for each node in the neuron  Tree
    std::unordered_map<NeuronNode*, NeuronNode*> connectPoint;
    std::unordered_map<NeuronNode*, double>       connectDist;

    for (NeuronNode* pNode : neuronTree) {
        double     minDist    = thresDist;   // 超过阈值则忽略
        NeuronNode* bestMatch = nullptr;

        for (BlockSimple* candidate : candidateGroups) {
            for (NeuronNode* cNode : *(candidate->pBlockNodeList)) {
                const double d = pNode->getDistanceTo(cNode);
                if (d >= minDist) continue;
                bestMatch = cNode;
                minDist   = d;
            }
        }
        if (bestMatch) {
            connectPoint[pNode] = bestMatch;
            connectDist[pNode]  = minDist;
        }
    }

    // Connect in order of distance from near to far
    while (!connectPoint.empty()) {
        // Find the point with the smallest connection distance
        NeuronNode* pNode      = nullptr;
        double      minDist    = std::numeric_limits<double>::max();

        for (const auto& kv : connectDist) {
            if (kv.second < minDist) {
                minDist = kv.second;
                pNode   = kv.first;
            }
        }
        NeuronNode* pConnect = connectPoint[pNode];
        connectPoint.erase(pNode);
        connectDist.erase(pNode);

        // Extract the branch to which the node belongs
        NodeList connectedSeg;
        if (!extractNeuronSegment(neuronTree, pNode, connectedSeg)) return false;

        // Adjust pNode to the root of the branch it belongs to
        if (pNode->parent && !setNewRoot(pNode)) return false;

        // Connect two nodes
        linkNodes(pNode, pConnect, fusionDist < minDist);

        // deal with other connectable points on the branch
        connectBranchNodes(connectedSeg, connectPoint, connectDist, fusionDist);

        // Add the successfully connected nodes to the result
        while (!connectedSeg.empty()) {
            NeuronNode* node = connectedSeg.front();
            node->type = pConnect->type;
            ConnectedSegs.addNode(node);
        }
    }
    return true;
}

// --------------------------------------------------------- //
// Search for 26 neighboring areas (generate new tracking targets)
void Neuron_Reconstruction::search26Neighbours(BlockSimple* cur, BlockSimpleList& newTargets, NodeList& tipList){
    std::vector<bool> flag(27, false);

    // Determine the direction to search based on the position of the peripheral point
    for (NeuronNode* tip : tipList) {
        int xx = 0, yy = 0, zz = 0;
        if      (std::abs(tip->x - cur->originX - marginSize) < 3)                          xx = -1;
        else if (std::abs(tip->x - cur->originX - cur->blockSizeX + marginSize) < 3)        xx =  1;
        if      (std::abs(tip->y - cur->originY - marginSize) < 3)                          yy = -1;
        else if (std::abs(tip->y - cur->originY - cur->blockSizeY + marginSize) < 3)        yy =  1;
        if      (std::abs(tip->z - cur->originZ - marginSize) < 3)                          zz = -1;
        else if (std::abs(tip->z - cur->originZ - cur->blockSizeZ + marginSize) < 3)        zz =  1;

        if (!xx && !yy && !zz) continue;

        if (xx)        flag[xx      + 13] = true;
        if (yy)        flag[yy * 3  + 13] = true;
        if (zz)        flag[zz * 9  + 13] = true;
        if (xx && yy)  flag[xx + yy * 3  + 13] = true;
        if (xx && zz)  flag[xx + zz * 9  + 13] = true;
        if (yy && zz)  flag[yy * 3 + zz * 9 + 13] = true;
        if (xx && yy && zz) flag[xx + yy * 3 + zz * 9 + 13] = true;
    }

    // Generate the neighborhood blocks that need to be processed
    for (int zz = -1; zz <= 1; ++zz) {
        if (zz == -1 && cur->originZ == 0) continue;
        if (zz ==  1 && cur->originZ + cur->blockSizeZ == imageSize[2]) continue;
        const V3DLONG zBS = (zz == 1) ? cur->blockSizeZ : blockSize;

        for (int yy = -1; yy <= 1; ++yy) {
            if (yy == -1 && cur->originY == 0) continue;
            if (yy ==  1 && cur->originY + cur->blockSizeY == imageSize[1]) continue;
            const V3DLONG yBS = (yy == 1) ? cur->blockSizeY : blockSize;

            for (int xx = -1; xx <= 1; ++xx) {
                if (xx == -1 && cur->originX == 0) continue;
                if (xx ==  1 && cur->originX + cur->blockSizeX == imageSize[0]) continue;
                if (!flag[xx + yy * 3 + zz * 9 + 13]) continue;
                const V3DLONG xBS = (xx == 1) ? cur->blockSizeX : blockSize;

                BlockSimple* nt = new BlockSimple();
                nt->originX = cur->originX + xx * (xBS - 2 * marginSize);
                nt->originY = cur->originY + yy * (yBS - 2 * marginSize);
                nt->originZ = cur->originZ + zz * (zBS - 2 * marginSize);


                nt->blockSizeX = (xx == 0 && blockSize > cur->blockSizeX)
                                     ? cur->blockSizeX : blockSize;
                nt->blockSizeY = (yy == 0 && blockSize > cur->blockSizeY)
                                     ? cur->blockSizeY : blockSize;
                nt->blockSizeZ = (zz == 0 && blockSize > cur->blockSizeZ)
                                     ? cur->blockSizeZ : blockSize;

                // boundary cropping
                auto clampBlock = [&]() {
                    if (nt->originX < 0) {
                        nt->blockSizeX = cur->originX + 2 * marginSize;
                        nt->originX    = 0;
                    }
                    if (nt->originY < 0) {
                        nt->blockSizeY = cur->originY + 2 * marginSize;
                        nt->originY    = 0;
                    }
                    if (nt->originZ < 0) {
                        nt->blockSizeZ = cur->originZ + 2 * marginSize;
                        nt->originZ    = 0;
                    }
                    if (nt->originX + nt->blockSizeX > imageSize[0])
                        nt->blockSizeX = imageSize[0] - nt->originX;
                    if (nt->originY + nt->blockSizeY > imageSize[1])
                        nt->blockSizeY = imageSize[1] - nt->originY;
                    if (nt->originZ + nt->blockSizeZ > imageSize[2])
                        nt->blockSizeZ = imageSize[2] - nt->originZ;
                };
                clampBlock();

                // 
                if (nt->blockSizeX < 2 * marginSize || nt->blockSizeY < 2 * marginSize || nt->blockSizeZ < 2 * marginSize) {
                    delete nt;
                    continue;
                }
                newTargets.push_back(nt);
            }
        }
    }
}

// --------------------------------------------------------- // 
// Free all node memory in the NodeList
void Neuron_Reconstruction::freeNodeList(NodeList& nl){
    while (!nl.empty()) {
        NeuronNode* node = nl.front();
        nl.pop_front();
        delete node;
    }
}

// Register processed blocks
NodeList* Neuron_Reconstruction::registerBlock(BlockSimple* currentTarget,const string& coordStr){
    auto it = blockMap.find(coordStr);
    if (it == blockMap.end()) {
        NodeList* pList        = new NodeList();
        currentTarget->pBlockNodeList = pList;
        blockMap[coordStr]     = currentTarget;
        blockList.push_back(currentTarget);
        return pList;
    } 
    else {
        delete currentTarget;
        return it->second->pBlockNodeList;
    }
}

// run Tracing Loop
void Neuron_Reconstruction::runTracingLoop(V3DPluginArgList& tracingArgsList)
{
    V3DLONG finalNeuronTreeSize = 0;
    V3DLONG traceCount          = static_cast<V3DLONG>(blockList.size());

    while (!allTargetList.empty() && finalNeuronTreeSize < MaxpointNUM)
    {
        std::cout << "[Info] Remaining target(s): " << allTargetList.size() << std::endl;

        // --- Load current block ---
        BlockSimple* currentTarget = allTargetList.front();
        allTargetList.pop_front();

        Block_boundary_adjust(currentTarget);

        const V3DLONG ox = currentTarget->originX;
        const V3DLONG oy = currentTarget->originY;
        const V3DLONG oz = currentTarget->originZ;
        const string coordStr = QString("x%1_y%2_z%3").arg(ox).arg(oy).arg(oz).toStdString();
        std::cout << "[Info] CurrentTarget: " << coordStr << std::endl;

        // ---  Neuron reconstruct in the current block ---
        NodeList blockNeuronTree;

        //Determine whether the image block has been reconstructed,
        const bool alreadyDone = readBlockSWC(QString::fromStdString(coordStr), ".tif_not_connect.swc", blockNeuronTree);
        
        //If not, extract the image block for reconstruction; otherwise read swc file to blockNeuronTree
        if (!alreadyDone) {
            if (!Block_Neuron_Reconstruct(currentTarget, tracingArgsList, blockNeuronTree, coordStr, ox, oy, oz)) {
                std::cerr << "[ERROR] Neuron block reconstruction failed." << std::endl;
                delete currentTarget;
                return;
            }
        }

        if (blockNeuronTree.isEmpty()) {
            delete currentTarget;
            currentTarget = nullptr;
            std::cout << "[Info] Empty target.\n" << std::endl;
            continue;
        }

        // --- Connect the current block to the global result ---
        NodeList connectedNeuronSegs;
        if (currentTarget->isStartBlock) {
            extractSomaNodes(blockNeuronTree, connectedNeuronSegs);
            currentTarget->isStartBlock = false;
        } 
        else {
            std::cout << "connection" << std::endl;
            BlockSimpleList candidateNeighbours;

            SearchNearBlock(currentTarget, candidateNeighbours);

            if (!connect(candidateNeighbours, blockNeuronTree, connectedNeuronSegs))
                std::cout << "[WARN] connect failed." << std::endl;
        }
       
        // --- Handle unconnected blocks ---
        const string notConnectFile = (tempFolderPath + "/").toStdString() + coordStr + ".tif_not_connect.swc";

        if (connectedNeuronSegs.isEmpty()) {
            delete currentTarget;
            saveSWCFile(notConnectFile, blockNeuronTree);
            std::cout << "[Info] Non-connected target.\n" << std::endl;
            freeNodeList(blockNeuronTree);
            continue;
        }

        // --- Search for new endpoints ---
        NodeList newTerminalPoints;
        collectLeafNodes(connectedNeuronSegs, newTerminalPoints);
        
        // --- 26 neighbours search ---
        search26Neighbours(currentTarget, allTargetList, newTerminalPoints);
        
        // --- record reconstructed block ---
        NodeList* pCurrentNodeList = registerBlock(currentTarget, coordStr);

        finalNeuronTreeSize += connectedNeuronSegs.size();
        while (!connectedNeuronSegs.empty())
        {
            NeuronNode* node = connectedNeuronSegs.front();
            connectedNeuronSegs.pop_front();   

            blockNeuronTree.removeOne(node);   
            pCurrentNodeList->addNode(node);   
        }

        saveSWCFile(notConnectFile, blockNeuronTree, true);
        freeNodeList(blockNeuronTree);

        // --- Save intermediate results regularly ---
        if (++traceCount % 10 == 0) {
            std::cout << "======== Periodic Save ========" << std::endl;
            double currentHours = globalTimer.nsecsElapsed() / 1e9 / 3600.0;
            saveFinalNeuronTree(finalSWCfile, currentHours);
        }

        std::cout << "[Info] Traced " << traceCount << " block(s). "
                  << "Neuron tree size: " << finalNeuronTreeSize << std::endl;
        
    }
}

// --------------------------------------------------------- // 
// Remove branches that are not connected to soma
void Neuron_Reconstruction::removeDisconnectedBranches(NodeList& nt, NeuronNode* soma){

    if (soma == nullptr) return;

    NodeList backs, fronts;
    for (NeuronNode* p : nt) {
        if (!p->children.empty()) continue;
        NeuronNode* bp = p;
        while (bp->parent) bp = bp->parent;
        if (bp == soma) continue;
        backs.push_back(p);
        fronts.push_back(bp);
    }
    for (NeuronNode* f : fronts) {
        if (!f->parent) continue;
        f->parent->children.removeOne(f);
        f->parent = nullptr;
    }
    for (NeuronNode* b : backs) {
        NeuronNode* cur = b;
        while (cur) { NeuronNode* nx = cur->parent; delete cur; cur = nx; }
    }
}

// Final Tree Pruning
bool Neuron_Reconstruction::pruneFinalBranch(int length, NodeList* NTree)
{
    if (!NTree) return false;

    NodeList backs, fronts;
    for (NeuronNode* node : *NTree) {
        if (!node->children.empty()) continue;
        NeuronNode* bp    = node;
        int         count = 0;
        while (bp->parent && bp->parent->children.size() < 2 && count < length) {
            bp = bp->parent;
            ++count;
        }
        if (count < length && bp->parent != nullptr){
            backs.push_back(node);
            fronts.push_back(bp);
        }
    }
    for (NeuronNode* f : fronts) {
        if (!f->parent) continue;
        f->parent->children.removeOne(f);
        f->parent = nullptr;
    }
    for (NeuronNode* b : backs) {
        NeuronNode* cur = b;
        while (cur) { NeuronNode* nx = cur->parent; delete cur; cur = nx; }
    }
    std::cout << "[Info] Final pruned " << fronts.size() << " branch(es)." << std::endl;
    return true;
}

// Final tree post-processing (pruning isolated branches)
bool Neuron_Reconstruction::adjustFinalNeuronTree(const QString& swcPath)
{
    const std::string filepath = swcPath.toStdString();
    std::ifstream ifs(filepath);
    if (!ifs)
    {
        std::cerr << "[ERROR] Cannot open " << filepath << std::endl;
        return false;
    }

    NodeList nt;
    std::unordered_map<int, NeuronNode*> marker_map;
    std::unordered_map<NeuronNode*, int> parid_map;

    NeuronNode* soma_node = nullptr;
    std::vector<NeuronNode*> root_nodes;

    //load swc
    for (std::string line; std::getline(ifs, line); )
    {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        int id = -1;
        int par_id = -1;

        NeuronNode* pNode = new NeuronNode;
        ss >> id >> pNode->type >> pNode->x >> pNode->y >> pNode->z >> pNode->radius >> par_id;

        if (ss.fail())
        {
            delete pNode;
            continue;
        }

        marker_map[id] = pNode;
        parid_map[pNode] = par_id;
        nt.addNode(pNode);

        // root/soma：parent == -1
        if (par_id == -1)
        {
            root_nodes.push_back(pNode);
            if (soma_node == nullptr)
                soma_node = pNode;   
        }
    }
    ifs.close();

    if (nt.isEmpty())
    {
        std::cerr << "[ERROR] Empty SWC file: " << filepath << std::endl;
        return false;
    }

    if (soma_node == nullptr)
    {
        std::cerr << "[ERROR] No root node (parent=-1) found in " << filepath << std::endl;
        return false;
    }

    if (root_nodes.size() > 1)
    {
        std::cout << "[Warning] Multiple root nodes found (" << root_nodes.size()
                  << "). Use the first root node as soma, and remove other disconnected components."
                  << std::endl;
    }

    // rebulid parent / children
    for (NeuronNode* pNode : nt)
    {
        int parid = parid_map[pNode];
        if (parid == -1) continue;

        auto it = marker_map.find(parid);
        if (it == marker_map.end()) continue;

        pNode->parent = it->second;
        pNode->parent->children.push_back(pNode);
    }

    // Remove branches that are not connected to soma
    removeDisconnectedBranches(nt, soma_node);

    // pruning
    pruneFinalBranch(min_branch_length, &nt);

    // Save results
    const std::string savepath = filepath + "_prune.swc";
    std::ofstream swcFile(savepath, std::ios::binary);
    if (!swcFile)
    {
        std::cerr << "[ERROR] Cannot open save file: " << savepath << std::endl;
        return false;
    }

    V3DLONG nums = 0;
    std::unordered_map<NeuronNode*, V3DLONG> ind;
    for (NeuronNode* node : nt)
        ind[node] = ++nums;

    swcFile << "# marker path " << qPrintable(markerPath) << "\n";
    swcFile << "# name " << filepath << "\n";
    swcFile << "# n, type, x, y, z, radius, parent\n";

    nums = 0;
    for (NeuronNode* node : nt)
    {
        V3DLONG parent_id = (node->parent && ind.count(node->parent)) ? ind[node->parent] : -1;
        swcFile << ++nums << " "
                << node->type << " "
                << node->x << " "
                << node->y << " "
                << node->z << " "
                << node->radius << " "
                << parent_id << "\n";
    }

    swcFile.close();
    std::cout << "[Info] Pruned SWC saved: " << savepath
              << ", node num = " << nums << std::endl;

    return true;
}

//===================================================================
// Main reconstruction function
//===================================================================
void Neuron_Reconstruction :: SINGLE_NEURON_RCONSTRUCT(){
    // obtain code current path
    // codePath = GetCWD();

    // step1: Initialize the output directory
    if (!initOutputDirectories()) {
        std::cerr << "[ERROR] Failed to initialize output directories." << std::endl;
        return;
    }
    
    // step2: Save Configuration
    configurationPath = outputFolderPath + "/configuration.yaml";
    if (!saveConfiguration(configurationPath)) {
        std::cerr << "[ERROR] Failed to save configuration." << std::endl;
        return;
    }

    // step3: Set tracking parameters
    V3DPluginArgList tracingArgsList;
    if (!initTracingArgs(tracingArgsList)) {
        std::cerr << "[ERROR] Failed to init tracing args." << std::endl;
        return;
    }

    // step4: Calculate edge size
    marginSize = static_cast<int>(margin_lamd * blockSize);
    cout<<"marginlamd: "<<margin_lamd<<"  "<<"marginSize: "<<marginSize<<endl;

    // step5: Calculate program runtime
    globalTimer.start();

    // step6: TeraFly initialization and load marker
     if (!initTeraFly()) return;

    // step7: Set start block 
    if (!initStartBlocks()) return;

    // step8: Main reconstruction Loop
    runTracingLoop(tracingArgsList);

    // step9: caculate time consumption
    double elapsedHours = globalTimer.nsecsElapsed() / 1e9 / 3600.0;

    // step10: Save the final result
    saveFinalNeuronTree(finalSWCfile, elapsedHours);

    adjustFinalNeuronTree(finalSWCfile);
    
    // step10: 
    cout << "========================================" << endl;
    cout<<"Neuron reconstruction finish"<<endl;
    cout<<"cost time: "<<elapsedHours<<" h"<<endl;
    cout << "========================================" << endl;
    return ;
}

//===================================================================
// I/O functions
//===================================================================
//Read block SWC file
bool Neuron_Reconstruction::readBlockSWC(const QString& coordinate, const QString& postfix, NodeList& nt, bool verbose)
{
    const string filePath = (tempFolderPath + "/" + coordinate + postfix).toStdString();

    std::ifstream ifs(filePath);
    if (!ifs) {
        if (verbose)
            std::cout << "[Info] Open " << filePath << " failed." << std::endl;
        return false;
    }

    nt.clear();
    std::unordered_map<int, NeuronNode*>  marker_map;
    std::unordered_map<NeuronNode*, int>  parid_map;

    for (string line; std::getline(ifs, line); ) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int id = -1, par_id = -1;
        ss >> id;
        if (ss.fail()) continue;

        NeuronNode* pNode = new NeuronNode;
        ss >> pNode->type >> pNode->x >> pNode->y >> pNode->z
           >> pNode->radius >> par_id;

        if (ss.fail()) {
            delete pNode;
            continue;
        }
        marker_map[id]  = pNode;
        parid_map[pNode] = par_id;
        nt.addNode(pNode);
    }

    // bulid relationship
    for (NeuronNode* pNode : nt) {
        const int parid = parid_map[pNode];
        if (parid == -1) continue;
        auto it = marker_map.find(parid);
        if (it == marker_map.end()) continue;
        pNode->parent = it->second;
        pNode->parent->children.push_back(pNode);
    }
    return true;
}

//Read Start Marker
bool Neuron_Reconstruction::readStartMarkers(const QString& markerFile)
{
    std::ifstream ifs(markerFile.toStdString(), std::ios::binary);
    if (!ifs) return false;

    for (string line; std::getline(ifs, line); ) {
        if (line.empty() || line[0] == '#') continue;
        
        // 
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);

        double ox = 0.0, oy = 0.0, oz = 0.0;
        ss >> ox >> oy >> oz;

        if (ss.fail())  continue; 

        BlockSimple* p = new BlockSimple;
        p->originX = static_cast<V3DLONG>(ox);
        p->originY = static_cast<V3DLONG>(oy);
        p->originZ = static_cast<V3DLONG>(oz);

        allTargetList.push_back(p);
    }
    return true;
}

// Save the final neural tree
bool Neuron_Reconstruction::saveFinalNeuronTree(const QString& savePath, double elapsedHours){
    if (QFileInfo(savePath).exists()) QFile(savePath).remove();

    const string filePath = savePath.toStdString();
    const string logPath  = outputFolderPath.toStdString() + "/log.txt";

    std::ofstream swcFile(filePath, std::ios::binary);
    std::ofstream logFile(logPath,  std::ios::binary);

    if (!swcFile) { std::cerr << "[ERROR] Cannot open: " << filePath << std::endl; return false; }
    if (!logFile) { std::cerr << "[ERROR] Cannot open: " << logPath  << std::endl; return false; }

    //
    V3DLONG nums  = 0;
    double minX = std::numeric_limits<double>::max(), maxX = std::numeric_limits<double>::lowest();
    double minY = minX, maxY = maxX, minZ = minX, maxZ = maxX;

    NeuronNode* soma_node = nullptr;
    std::unordered_map<NeuronNode*, V3DLONG> ind;

    for (BlockSimple* pBlock : blockList) {
        for (NeuronNode* node : *(pBlock->pBlockNodeList)) {
            minX = std::min(minX, node->x); maxX = std::max(maxX, node->x);
            minY = std::min(minY, node->y); maxY = std::max(maxY, node->y);
            minZ = std::min(minZ, node->z); maxZ = std::max(maxZ, node->z);
            //if (node->radius > rmax) { rmax = node->radius; soma_node = node; }
            if (node->parent == nullptr && soma_node == nullptr)
                soma_node = node;
            ind[node] = ++nums;
        }
    }

    // set the root point to soma node
    if (soma_node && soma_node->parent)
        if (!setNewRoot(soma_node)) return false;

    // 
    swcFile << "# marker path " << qPrintable(markerPath) << "\n"
            << "# name "        << filePath               << "\n"
            << "# n, type, x, y, z, radius, parent\n"
            << "# x range: " << (maxX - minX)
            << ", y range: " << (maxY - minY)
            << ", z range: " << (maxZ - minZ) << "\n";

    logFile << "# Image path: " << qPrintable(imagePath) << "\n"
            << "# Marker path: " << qPrintable(markerPath) << "\n"
            << "# Output path: " << qPrintable(outputFolderPath) << "\n"
            << "# Trace method: " << qPrintable(traceMethod) << "\n"
            << "# Block size: " << blockSize << "\n"
            << "# Margin size: " << marginSize << "\n"
            << "# Total blocks traced: " << blockList.size() << "\n"
            << "# Total nodes: " << nums << "\n"
            << "# Total time: " << elapsedHours <<" h"<< "\n"
            << "# Soma: "
            << (soma_node ? QString("(%1, %2, %3) r=%4")
                               .arg(soma_node->x).arg(soma_node->y)
                               .arg(soma_node->z).arg(soma_node->radius)
                               .toStdString()
                          : "not found")
            << "\n"
            << "# X range: [" << minX << ", " << maxX << "] span=" << (maxX - minX) << "\n"
            << "# Y range: [" << minY << ", " << maxY << "] span=" << (maxY - minY) << "\n"
            << "# Z range: [" << minZ << ", " << maxZ << "] span=" << (maxZ - minZ) << "\n"
            << "# ================================================\n"
            << "# sub blocks: \n"
            << "# @ blockX blockY blockZ blockSizeX blockSizeY blockSizeZ\n"
            << "# ================================================\n";


    // 
    nums = 0;
    for (BlockSimple* pBlock : blockList) {
        logFile << "@ " << pBlock->originX << " " << pBlock->originY << " " << pBlock->originZ
                << " "  << pBlock->blockSizeX << " " << pBlock->blockSizeY << " "
                << pBlock->blockSizeZ << "\n";

        for (NeuronNode* node : *(pBlock->pBlockNodeList)) {
            const V3DLONG pid = (node->parent && ind.count(node->parent)) ? ind.at(node->parent) : -1;
            const auto line = QString("%1 %2 %3 %4 %5 %6 %7\n")
                                  .arg(++nums).arg(node->type)
                                  .arg(node->x).arg(node->y).arg(node->z)
                                  .arg(node->radius).arg(pid);
            swcFile << line.toStdString();
            logFile << line.toStdString();
        }
    }

    std::cout << "[Info] Saved " << nums << " nodes to " << filePath << std::endl;
    return true;
}

// ---------------------------------------------- Global function ------------------------------------------------------
//===================================================================
// D_LSNARS 
//===================================================================
QString cleanString(QString str) {
    return str.remove('\'').remove('\"').trimmed();
}

bool D_LSNARS(V3DPluginCallback2& callback,
              V3DPluginArgList& args,
              const string& inputImage,
              const QString &pythoncodepath,
              bool isStartBlock)
{
    auto* pArgs = static_cast<vector<char*>*>(args.back().p);
    if (!pArgs || pArgs->size() < 3) return false;

    const QString pyEnv   = cleanString(QString::fromUtf8((*pArgs)[0]));
    const QString cfgPath = cleanString(QString::fromUtf8((*pArgs)[1]));
    const QString outDir  = cleanString(QString::fromUtf8((*pArgs)[2]));
    const QString inPath  = cleanString(QString::fromStdString(inputImage));
    const QString pyCodeDir = cleanString(pythoncodepath);

    // 固定代码路径（建议移至配置）
    //const QString codePath =
    //    "F:/neuron_reconstruction_system/D_LSNARS_test/main.py";
    const QString codePath = QDir(pyCodeDir).filePath("main.py");

    const QString command =
        QString("conda activate %1 && python %2 -i %3 -c %4 -s %5")
            .arg(pyEnv, codePath, inPath, cfgPath,
                 isStartBlock ? "1" : "0");

#ifdef _WIN32
    int ret = ::system(qPrintable(command));
    return ret == 0;
#else
    std::cerr << "[ERROR] D_LSNARS: unsupported platform." << std::endl;
    return false;
#endif
}

//===================================================================
// Boundary interpolation
//===================================================================
NeuronNode *interpolateNodeOnBoundary(NeuronNode *src, double boundary, Direction direction) {

    if (!src || !src->parent) return nullptr;

    NeuronNode* oldParent = src->parent;
    NeuronNode* iNode     = new NeuronNode();

    const double dx = oldParent->x - src->x;
    const double dy = oldParent->y - src->y;
    const double dz = oldParent->z - src->z;

    double scale = 0.0;
    switch (direction) {
        case Direction::LeftSide:
        case Direction::RightSide:
            scale = (dx != 0.0) ? (boundary - src->x) / dx : 0.0;
            break;
        case Direction::UpSide:
        case Direction::DownSide:
            scale = (dy != 0.0) ? (boundary - src->y) / dy : 0.0;
            break;
        case Direction::OutSide:
        case Direction::InSide:
            scale = (dz != 0.0) ? (boundary - src->z) / dz : 0.0;
            break;
    }

    iNode->x = src->x + dx * scale;
    iNode->y = src->y + dy * scale;
    iNode->z = src->z + dz * scale;
    iNode->radius = (oldParent->radius + src->radius) * 0.5;
    iNode->type   = oldParent->type;

    //Rebuilds the relationship 
    oldParent->children.removeOne(src);
    oldParent->children.push_back(iNode);
    iNode->parent = oldParent;
    src->parent   = iNode;
    iNode->children.push_back(src);

    return iNode;
}

//===================================================================
// Resampling (calling plugin)
//===================================================================
bool resample(V3DPluginCallback2& callback, const string& swcfile, string stepLen, const string& resampledSwcFile)
{
    V3DPluginArgList in, out;
    vector<char*> infile{const_cast<char*>(swcfile.data())};
    vector<char*> inpara{const_cast<char*>(stepLen.data())};
    vector<char*> output{const_cast<char*>(resampledSwcFile.data())};

    in.push_back(V3DPluginArgItem());  in.back().p  = &infile;
    in.push_back(V3DPluginArgItem());  in.back().p  = &inpara;
    out.push_back(V3DPluginArgItem()); out.back().p = &output;

    return callback.callPluginFunc("resample_swc", "resample_swc", in, out);
}

//===================================================================
// Get the current working directory
//===================================================================
char* GetCWD()
{
    char* str = ::getcwd(nullptr, 0);
    if (!str) {
        perror("getcwd error");
        return nullptr;
    }
    printf("dll path: %s\n", str);
    // 注意：调用方负责 free(str)
    return str;
}

//===================================================================

//===================================================================

/*
bool whole_brain_soma_location(V3DPluginCallback2 &callback, V3DPluginArgList &args){
    QString pythonEnvironment((static_cast<vector<char *> *>(args.front().p))->at(0)),
    inputPath((static_cast<vector<char *> *>(args.front().p))->at(1)),
    configurationPath((static_cast<vector<char *> *>(args.front().p))->at(2)),
    outputFolderPath((static_cast<vector<char *> *>(args.front().p))->at(3));
    //cout<<"pythonEnvironment: "<<pythonEnvironment.toStdString()<<endl;
    //cout<<"image file path: "<<inputPath.toStdString()<<endl;
    //cout<<"configurationPath"<<configurationPath.toStdString()<<endl;
    //cout<<"outputFolderPath"<<outputFolderPath.toStdString()<<endl;

    QString codepath = "F:\\neuron_reconstruction_system\\D_LSNARS\\whole_brain_somas_detection\\src\\python\\Soma_location.py";
    QString command = QString("conda activate %1 && python ");
    command.append(QString(codepath)).append(QString(" -i %2 -c %3 -o %4"));
#ifdef _WIN32
    int ret = system(qPrintable(command.arg(pythonEnvironment).arg(inputPath).arg(configurationPath).arg(outputFolderPath)));
#endif
    if (ret!=0) return FALSE;
    return TRUE;               
}
*/
/*
bool resample(const string &swcfile, string stepLen, const string &resampledSwcFile){
    NodeList Ntree;
    readSWCtoNodeList(swcfile,Ntree);

    return true;
}
*/

