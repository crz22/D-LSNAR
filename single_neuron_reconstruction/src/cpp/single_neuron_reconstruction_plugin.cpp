/* single_neuron_reconstruction_plugin.cpp
 * This is a test plugin, you can use it as a demo.
 * 2024-7-22 : by Runze Chen
 */
 
#include "v3d_message.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <io.h>
#include "single_neuron_reconstruction_plugin.h"
using namespace std;

Q_EXPORT_PLUGIN2(single_neuron_reconstruction, TestPlugin);
 
QStringList TestPlugin::menulist() const
{
	return QStringList() 
		<<tr("reconstruction set")
		<<tr("about");
}

QStringList TestPlugin::funclist() const
{
	return QStringList()
		<<tr("func1")
		<<tr("func2")
		<<tr("help");
}

void TestPlugin::domenu(const QString &menu_name, V3DPluginCallback2 &callback, QWidget *parent)
{
	if (menu_name == tr("reconstruction set"))
	{
		auto dialog = new Reconstruction_Dialog(callback,parent);
		dialog->exec();
		//v3d_msg("To be implemented.");
	}
	else if (menu_name == tr("menu2"))
	{
		v3d_msg("To be implemented.");
	}
	else
	{
		v3d_msg(tr("This is a test plugin, you can use it as a demo.. "
			"Developed by Runze Chen, 2024-7-22"));
	}
}

bool TestPlugin::dofunc(const QString & func_name, const V3DPluginArgList & input, V3DPluginArgList & output, V3DPluginCallback2 & callback,  QWidget * parent)
{
	vector<char*> infiles, outfiles, params;
	if(input.size() >= 1) infiles = *((vector<char*> *)input.at(0).p);
	if(input.size() >= 2) params = *((vector<char*> *) input.at(1).p);

	if(output.size() >= 1) outfiles = *((vector<char*> *)output.at(0).p);
	

	if (func_name == tr("reconstruction"))
	{	
		//cout<<infiles<<" "<<marker_files<<" "<<pytorch_path<<" "<<outfiles<<endl;
		QString image_path, marker_path, pytorch_path, save_path, configuration_path;
		image_path = QString::fromUtf8(infiles.at(0));
		marker_path = QString::fromUtf8(infiles.at(1));
		pytorch_path = QString::fromUtf8(infiles.at(2));
		configuration_path = QString::fromUtf8(params.at(0));

		save_path = QString::fromUtf8(outfiles.at(0));
		
		cout << "image_path: "<<image_path.toStdString()<<endl;
		cout << "marker_files: "<<marker_path.toStdString()<<endl;
		cout << "pytorch_path: "<<pytorch_path.toStdString()<<endl;
		cout << "configuration_path: "<<configuration_path.toStdString()<<endl; 

		cout << "outfiles: "<<save_path.toStdString()<<endl;

		cout<<" ################  cfgs params  ################# "<<endl;
		std::vector<std::pair<std::string, std::string>> cfgs = LoadConfig(configuration_path.toStdString());
		for (const auto& kv : cfgs)
			{
				std::cout << kv.first
						<< " : "
						<< kv.second
						<< std::endl;
			}

		cout<<" ############################################## "<<endl;

		neuron_reconstruction = new Neuron_Reconstruction(callback);

		neuron_reconstruction->imagePath = image_path;
		neuron_reconstruction->markerPath = marker_path;
		neuron_reconstruction->pytorchPath = pytorch_path;
		
		//neuron_reconstruction->configurationPath = configuration_path;

		neuron_reconstruction->outputFolderPath = save_path;
		
		neuron_reconstruction->Cfgs = cfgs;

		neuron_reconstruction->pythonCodePath = QString::fromStdString(GetCfg(cfgs, "Python_code_path"));;
		neuron_reconstruction->traceMethod = QString::fromStdString(GetCfg(cfgs, "rec_model_name"));
		neuron_reconstruction->segmentMethod = QString::fromStdString(GetCfg(cfgs, "seg_model_name"));
		neuron_reconstruction->blockSize = atoi(GetCfg(cfgs, "blockSize").c_str());
		neuron_reconstruction->margin_lamd = atof(GetCfg(cfgs, "margin_lamd").c_str());;
		neuron_reconstruction->MaxpointNUM = atoi(GetCfg(cfgs, "MaxpointNUM").c_str());
		neuron_reconstruction->min_branch_length = atoi(GetCfg(cfgs, "min_branch_length").c_str());

		//neuron_reconstruction->node_step = 2;
		//neuron_reconstruction->branch_MAXL = 1000;
		//neuron_reconstruction->Angle_T = 1.57;  //1.047
		//neuron_reconstruction->Lamd = 4;

		neuron_reconstruction->SINGLE_NEURON_RCONSTRUCT();
		delete neuron_reconstruction;
		
		cout<<"finished"<<endl;

	}
	else if (func_name == tr("func2"))
	{
		v3d_msg("To be implemented.");
	}
	else if (func_name == tr("help"))
	{
		v3d_msg("To be implemented.");
	}
	else return false;

	return true;
}


