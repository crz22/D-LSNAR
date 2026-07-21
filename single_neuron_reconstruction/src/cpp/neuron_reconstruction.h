#ifndef _NEURON_RECONSTRUCTION_
#define _NEURON_RECONSTRUCTION_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <unordered_map>    
#include <unordered_set>    
#include <algorithm>        
#include <limits>           
#include <direct.h>

#include <QtGui>
#include <QtCore/QVariant>
#include <QDateTime>
#include <QElapsedTimer>    

#include "v3d_interface.h"
#include "basic_surf_objs.h"
#include "io2.h"
using namespace std;
enum Direction{LeftSide = 0, RightSide, UpSide, DownSide, OutSide, InSide};

class Neuron_Reconstruction: public QObject
{
    Q_OBJECT

public:
    QString imagePath;
    QString markerPath;
    QString outputFolderPath;
    QString pytorchPath;
    QString pythonCodePath;

    QString traceMethod;
    QString segmentMethod;
    int blockSize;
    float margin_lamd;
    long long MaxpointNUM;
    int marginSize;
    int node_step;
    int branch_MAXL;
    float   Angle_T;
    float   Lamd;
    int min_branch_length;

    bool terafly = true;

    BlockSimpleList blockList;
    BlockSimpleList allTargetList;

    std::vector<std::pair<std::string, std::string>> Cfgs;

private:
    V3DPluginCallback2& callback;
    QString tempFolderPath;
    QString configurationPath;
    QString finalSWCfile;
    //string codePath;

    vector<string> parasString;
    std::vector<char*>        paras;      
    std::vector<char*>       subImageVec; 
    string currentSubImagePath;

    V3DLONG* imageSize = new V3DLONG[4]();

    int somatype = 2;  //points connect with soma
    
    unordered_map<string, BlockSimple*> blockMap;

    QElapsedTimer globalTimer;   //

public:
    explicit Neuron_Reconstruction(V3DPluginCallback2& cb);
    ~Neuron_Reconstruction()
    {
        delete[] imageSize;
    }
    
    void SINGLE_NEURON_RCONSTRUCT();
    
    //===================================================================
    bool readStartMarkers(const QString &markerFile);
    bool readBlockSWC(const QString &coordinate, const QString &postfix, NodeList &nt, bool verbose = FALSE);
    bool saveFinalNeuronTree(const QString &savePath, double elapsedHours = -1.0);

    //===================================================================
    bool initOutputDirectories();
    bool saveConfiguration(const QString &savePath);
    bool initTracingArgs(V3DPluginArgList& tracingArgsList);
    bool initTeraFly();
    bool fetchAndReadTeraFlyMarker(const string& markerFilePath);
    bool initStartBlocks();

    //===================================================================
    void runTracingLoop(V3DPluginArgList& tracingArgsList);

    void Block_boundary_adjust(BlockSimple *pBlock);
    bool Block_Neuron_Reconstruct(BlockSimple *currentTarget,V3DPluginArgList &args, NodeList &blockNeuronTree,
                                            const string &coordinateString,V3DLONG OriginX, V3DLONG OriginY, V3DLONG OriginZ);
    void SearchNodesOnBoundary(BlockSimple *Target, NodeList &nt, int margin);
    void findTips(NodeList &nt, V3DLONG start, V3DLONG end, Direction direction);
    bool pruneTinyBranch(int length, NodeList *nodeList, BlockSimple *blockSimple);

    void extractSomaNodes(NodeList& src, NodeList& dst);
    void collectLeafNodes(const NodeList& src, NodeList& dst);

    void SearchNearBlock(BlockSimple *centralblock, BlockSimpleList &candidateNeighbours);
    bool connect(BlockSimpleList &candidateGroups, NodeList &neuronTree, NodeList &connectedSegs, double thresDist = 5.0); 
    bool extractNeuronSegment(NodeList &nt, NeuronNode *pNode, NodeList &neuronSeg);
    bool setNewRoot(NeuronNode *newRoot);
    void linkNodes(NeuronNode* pNode, NeuronNode* pConnect, bool useEdge);
    void connectBranchNodes(NodeList& branch,std::unordered_map<NeuronNode*, NeuronNode*>& connectPoint,std::unordered_map<NeuronNode*, double>&connectDist, double fusionDist);
    bool findNodeInBranch(NeuronNode *branchNode,NeuronNode *searchNode);

    void search26Neighbours(BlockSimple *currentTarget, BlockSimpleList &newTargetList, NodeList &tipList); 

    void freeNodeList(NodeList& nl);
    NodeList* registerBlock(BlockSimple* currentTarget,const string& coordStr);

    bool adjustFinalNeuronTree(const QString &swcPath);
    void removeDisconnectedBranches(NodeList& nt, NeuronNode* soma);
    bool pruneFinalBranch(int length, NodeList *NTree);

};

// ---------------------------------------------- Global function ------------------------------------------------------
char* GetCWD();

bool resample(V3DPluginCallback2 &callback, const string &swcfile, string stepLen, const string &resampledSwcFile);

bool D_LSNARS(V3DPluginCallback2& callback, V3DPluginArgList& args, const string& inputImage, const QString &pythoncodepath, bool isStartBlock);

NeuronNode *interpolateNodeOnBoundary(NeuronNode *src, double boundary, Direction direction);

#endif //_NEURON_RECONSTRUCTION_

