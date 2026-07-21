#include "data_definition.h"

void NodeList::addNode(NeuronNode* const &node) {
    push_back(node);
    if (node->pBelongToNodeList != nullptr) 
        node->pBelongToNodeList->removeOne(node);
    node->pBelongToNodeList = this;
}

NodeList &NodeList::operator=(const NeuronTree &neuronTree) {
    while (!this->isEmpty()) {
        delete this->front();
    }
    QHash<V3DLONG, NeuronNode*> hashNode;
    vector<V3DLONG> parent;
    hashNode[-1] = nullptr;
    for (const auto& swc: neuronTree.listNeuron) 
    {
        auto n = new NeuronNode;
        n->type = swc.type;
        n->x = swc.x;
        n->y = swc.y;
        n->z = swc.z;
        n->radius = swc.r;
        hashNode[swc.n] = n;
        parent.push_back(swc.pn);
        this->addNode(n);
    }
    int num = 0;
    for (auto node: *this) 
    {
        node->parent = hashNode[parent[num++]];
    }
    return *this;
}

void NodeList::convertToNeuronTree(NeuronTree &neuronTree) {
    QHash <NeuronNode*, int> hashNode;
    int num = 0;
    for (auto node: *this) {
        hashNode[node] = num++;
    }
    num = 0;
    for (auto node: *this) {
        NeuronSWC swc;
        swc.n = num;
        swc.type = node->type;
        swc.x = node->x;
        swc.y = node->y;
        swc.z = node->z;
        swc.r = node->radius;
        swc.pn = (node->parent)? hashNode[node->parent]: -1;
        neuronTree.listNeuron.append(swc);
        neuronTree.hashNeuron.insert(num, num);
        num++;
    }
}
