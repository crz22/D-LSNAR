#ifndef __DIALOG_H__
#define __DIALOG_H__

#include <QtGui>
#include "v3d_interface.h"
#include <QWidget>
#include <iostream>
#include <QApplication>
#include <QDateTime>
//#include "surf_objs.h"
#include "basic_surf_objs.h"
#include <qtpropertybrowser.h>
#include <qtpropertymanager.h>
#include <qtvariantproperty.h>
#include <qttreepropertybrowser.h>
#include "neuron_reconstruction.h"


#define PI 3.1415926
using namespace std;

enum {SPEDNR=0, APP2};
enum {DTANET=0, UNET3D};

//Dialog functions
class Reconstruction_Dialog : public QDialog
{
	Q_OBJECT

public:
    QGridLayout *OverallLayout;
    
    //file input and output
    QGroupBox *groupBox_filepath;
    QGridLayout *Layout_filepath;
    
    QHBoxLayout *HBox_input_path;
    QLabel *input_path_label;
    QLineEdit *input_path_text;
    QPushButton *input_path_button;

    QHBoxLayout *HBox_marker_path;
    QLabel *marker_path_label;
    QLineEdit *marker_path_text;
    QPushButton *marker_path_button;
    
    QHBoxLayout *HBox_save_path;
    QLabel *save_path_label;
    QLineEdit *save_path_text;
    QPushButton *save_path_button;

    QHBoxLayout *HBox_pytorch_path;
    QLabel *pytorch_path_label;
    QLineEdit *pytorch_path_text;
    QPushButton *pytorch_path_button;

    QString save_path_value;
    QString input_path_value;
    QString marker_path_value;
    QString pytorch_path_value;
    
    //parameter
    QGroupBox *groupBox_parameters;
    QGridLayout *Layout_parameters;

    QHBoxLayout *HBox_configuration_path;
    QLabel *configuration_path_label;
    QLineEdit *configuration_path_text;
    QPushButton *configuration_path_button;

    QString configuration_path_value;

    /* */
    QLabel    *configuration_params_label;
    QTextEdit *configuration_params_text;


    /**/
    //start button
    QPushButton *Start_button;
    Neuron_Reconstruction *neuron_reconstruction;


private:
    V3DPluginCallback2& _callback;
    void SETfilepath(V3DPluginCallback2& callback);
    void SETparameter();

    std::vector<std::pair<std::string, std::string>> cfgs;   
    void reloadCfgsAndShow();

public:
    explicit Reconstruction_Dialog(V3DPluginCallback2& callback, QWidget* parent = nullptr);
    ~Reconstruction_Dialog();   

private slots:
    //private slots
    
    void Select_input_path();
    void Select_marker_path();
    void Select_save_path();
    void Select_pytorch_path();

    void update_input_path();
    void update_marker_path();
    void update_save_path();
    void update_pytorch_path();

    /**/
    void Select_configuration_path();
    void update_configuration_path();

    void onConfigPathButtonClicked();
    void onConfigPathEditingFinished();


    /**/ 
    void Start_neuron_reconstruction();   
};

#endif //__DIALOG_H__


