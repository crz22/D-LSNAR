#ifndef _DATA_DEFINITION_
#define _DATA_DEFINITION_

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <QtGui>
#include <QtCore>
#include <QDateTime>
#include <QApplication>
#include <unordered_set>
#include <unordered_map>

#include "v3d_interface.h"
#include "basic_surf_objs.h"

using namespace std;

struct NeuronNode
{
    int type;
    double x;
    double y;
    double z;
    double radius;
    NeuronNode* parent;
    QList<NeuronNode*> children;
    QList<NeuronNode*>* pBelongToNodeList;
    
    NeuronNode(): type(0), x(0), y(0), z(0), radius(0), parent(nullptr), pBelongToNodeList(nullptr){}
    NeuronNode(int x_, int y_, int z_): NeuronNode() { x = x_, y = y_, z = z_;}
    NeuronNode(int type_, int x_, int y_, int z_, int radius_, NeuronNode* parent_, QList<NeuronNode*>& children_, QList<NeuronNode*>* blockPointer_):
        type(type_), x(x_), y(y_), z(z_), radius(radius_), parent(parent_), children(children_), pBelongToNodeList(blockPointer_){}
    NeuronNode(const NeuronNode& other): type(other.type), x(other.x), y(other.y), z(other.z), parent(other.parent), children(other.children), pBelongToNodeList(other.pBelongToNodeList){}
    NeuronNode& operator=(const NeuronNode& other)
    {
        if (&other == this) return *this;
        this->type = other.type;
        this->x = other.x;
        this->y = other.y;
        this->z = other.z;
        this->parent = other.parent;
        this->children = other.children;
        this->pBelongToNodeList = other.pBelongToNodeList;
        return *this;
    }
    ~NeuronNode()
    {
        if (parent) parent->children.removeOne(this);
        for (NeuronNode* child: children) child->parent = nullptr;
        if (pBelongToNodeList) pBelongToNodeList->removeOne(this);
    }
    double getDistanceTo(const NeuronNode* other) const {return sqrt(pow(this->x - other->x, 2) + pow(this->y - other->y, 2) + pow(this->z - other->z, 2));}
//    double getGeometryDistanceBetween(const Node* other, const Image4DSimple* image);
};

class NodeList: public QList<NeuronNode*>
{
public:
    NodeList(): QList<NeuronNode *>() {}
    //explicit NodeList(const NeuronTree& neuronTree, V3DLONG originX = 0, V3DLONG originY = 0, V3DLONG originZ = 0);
    void addNode(NeuronNode* const& node);
    void extractAllNodes(NodeList& other) {
        while (!other.empty()) 
            this->addNode(other.front());
    }
    //explicit operator NeuronTree();
    NodeList& operator=(const NeuronTree& neuronTree);
    void convertToNeuronTree(NeuronTree& neuronTree);
private:
    NodeList(const NodeList&); // copy constructor is not supported
    NodeList& operator=(const NodeList&); // operator= is not supported
};

//typedef QList<NeuronNode*> NodeList;
struct BlockSimple
{
    bool isStartBlock;
    V3DLONG originX, originY, originZ, blockSizeX, blockSizeY, blockSizeZ;
    NodeList* pBlockNodeList;
    BlockSimple():isStartBlock(false), originX(-1), originY(-1), originZ(-1), blockSizeX(-1), blockSizeY(-1), blockSizeZ(-1), pBlockNodeList(nullptr) {};
    BlockSimple(const BlockSimple&);
    //BlockSimple& operator=(const BlockSimple&);
    ~BlockSimple()
    {
        if (pBlockNodeList != nullptr)
        {
            while (!pBlockNodeList->empty()) {
                delete pBlockNodeList->front();
            }
            delete pBlockNodeList;
        }
    }
};

typedef QList<BlockSimple*> BlockSimpleList;

// root node with parent 0
struct MyMarker
{
	double x;
	double y;
	double z;
	union
	{
    #ifdef __SET_MARKER_DEGREE__
		double degree;
    #endif
		double radius;
	};
	int type;
	MyMarker* parent;
	MyMarker(){x=y=z=radius=0.0; type = 3; parent=0;}
	MyMarker(double _x, double _y, double _z) {x = _x; y = _y; z = _z; radius = 0.0; type = 3; parent = 0;}
	MyMarker(const MyMarker & v){x=v.x; y=v.y; z=v.z; radius = v.radius; type = v.type; parent = v.parent;}
	//MyMarker(const MyPoint & v){x=v.x; y=v.y; z=v.z; radius = 0.0; type = 3; parent = 0;}
	//MyMarker(const MYXYZ & v){x=v.x; y=v.y; z=v.z; radius = 0.0; type = 3; parent = 0;}

	double & operator [] (const int i) {
		assert(i>=0 && i <= 2);
		return (i==0) ? x : ((i==1) ? y : z);
	}

	bool operator<(const MyMarker & other) const{                                                                 
		if(z > other.z) return false;
		if(z < other.z) return true;
		if(y > other.y) return false;
		if(y < other.y) return true;
		if(x > other.x) return false;
		if(x < other.x) return true;    
		return false;
	}
	bool operator==(const MyMarker & other) const{                                                                 
		return (z==other.z && y==other.y && x==other.x);
	}
	bool operator!=(const MyMarker & other) const{                                                                 
		return (z!=other.z || y!=other.y || x!=other.x);
	}

	long long ind(long long sz0, long long sz01)
	{
		return ((long long)(z+0.5) * sz01 + (long long)(y+0.5)*sz0 + (long long)(x+0.5));
	}
};

#endif //_DATA_DEFINITION_