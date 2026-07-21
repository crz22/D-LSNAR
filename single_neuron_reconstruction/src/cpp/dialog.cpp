#include "dialog.h"

Reconstruction_Dialog ::Reconstruction_Dialog(V3DPluginCallback2 &callback, QWidget *parent):
     _callback(callback),
    OverallLayout(new QGridLayout()),
    groupBox_filepath(new QGroupBox("File path",this)),
    groupBox_parameters(new QGroupBox("parameter set",this))
{
    //window set
    this->setWindowTitle("single neuron reconstruction");
    this->resize(400,100);
    this->setWindowFlags(windowFlags() | Qt::WindowMaximizeButtonHint);
    this->setWindowFlags(windowFlags() | Qt::WindowMinimizeButtonHint);

    //Set file path
    SETfilepath(callback);
    
    //Set neuron reconstruct parameters
    SETparameter();

    /**/
    Start_button = new QPushButton("Start");
    connect(Start_button,SIGNAL(clicked()),this,SLOT(Start_neuron_reconstruction()));

    OverallLayout->addWidget(groupBox_filepath,0,0);
    OverallLayout->addWidget(groupBox_parameters,1,0);
    OverallLayout->addWidget(Start_button,2,0);

    this->setLayout(OverallLayout);

    cout<<"initial finish"<<endl;

}

/*-----------------------------------------------------------------------*/
void Reconstruction_Dialog :: SETfilepath(V3DPluginCallback2& callback){
    //Set file path
    Layout_filepath = new QGridLayout();

    /**/
    HBox_input_path = new QHBoxLayout();
    input_path_label = new QLabel("TeraFly file path: ");
    input_path_text = new QLineEdit();
    input_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");

    v3dhandle curWin = callback.currentImageWindow();
    if (curWin) {
        QString imageName = callback.getImageName(curWin);
        if (!imageName.endsWith(".tif")) {
            input_path_text->setText(callback.getPathTeraFly());
            //teraflyChecker->setChecked(Qt::CheckState::Checked);
        } 
        else input_path_text->setText(imageName);
        input_path_value = input_path_text->text();
        input_path_text->setEnabled(false);
        input_path_button->setEnabled(false);
    }

    HBox_input_path->addWidget(input_path_label);
    HBox_input_path->addWidget(input_path_text);
    HBox_input_path->addWidget(input_path_button);
    
    /**/
    HBox_marker_path = new QHBoxLayout();
    marker_path_label = new QLabel("Marker file path: ");
    marker_path_text = new QLineEdit();
    marker_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");

    HBox_marker_path->addWidget(marker_path_label);
    HBox_marker_path->addWidget(marker_path_text);
    HBox_marker_path->addWidget(marker_path_button);

    /**/
    HBox_save_path = new QHBoxLayout();
    save_path_label = new QLabel("Save file path:    ");
    save_path_text = new QLineEdit();
    //save_path_text->setText("");
    save_path_value = save_path_text->text();
    save_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");

    HBox_save_path->addWidget(save_path_label);
    HBox_save_path->addWidget(save_path_text);
    HBox_save_path->addWidget(save_path_button);

    /**/
    HBox_pytorch_path = new QHBoxLayout();
    pytorch_path_label = new QLabel("Pytorch path: ");
    pytorch_path_text = new QLineEdit();
    pytorch_path_text->setText("D:/ruanjian/minicoda/envs/pytorch");
    pytorch_path_value = pytorch_path_text->text();
    pytorch_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");

    HBox_pytorch_path->addWidget(pytorch_path_label);
    HBox_pytorch_path->addWidget(pytorch_path_text);
    HBox_pytorch_path->addWidget(pytorch_path_button);
    
    /**/
    Layout_filepath->addLayout(HBox_input_path,0,0);
    Layout_filepath->addLayout(HBox_marker_path,1,0);
    Layout_filepath->addLayout(HBox_save_path,2,0);
    Layout_filepath->addLayout(HBox_pytorch_path,3,0);
    groupBox_filepath->setLayout(Layout_filepath);

    connect(input_path_button, SIGNAL(clicked()), this, SLOT(Select_input_path()));
    connect(marker_path_button, SIGNAL(clicked()), this, SLOT(Select_marker_path()));
    connect(save_path_button, SIGNAL(clicked()), this, SLOT(Select_save_path()));
    connect(pytorch_path_button, SIGNAL(clicked()), this, SLOT(Select_pytorch_path()));

    connect(save_path_text, SIGNAL(editingFinished()), this, SLOT(update_save_path()));
    connect(input_path_text, SIGNAL(editingFinished()), this, SLOT(update_input_path()));
    connect(marker_path_text, SIGNAL(editingFinished()), this, SLOT(update_marker_path()));
    connect(pytorch_path_text, SIGNAL(editingFinished()), this, SLOT(update_pytorch_path()));
}

/*-----------------------------------------------------------------------*/
void Reconstruction_Dialog :: SETparameter(){
    //Set neuron reconstruct parameters
    Layout_parameters = new QGridLayout();
    
    /* */
    HBox_configuration_path = new QHBoxLayout();
    configuration_path_label = new QLabel("Params set yaml path: ");
    configuration_path_text = new QLineEdit();
    configuration_path_text->setText("code_download_dir/src/python/setup.yaml");
    configuration_path_value = configuration_path_text->text();
    configuration_path_button = new QPushButton(QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton), "");

    HBox_configuration_path->addWidget(configuration_path_label);
    HBox_configuration_path->addWidget(configuration_path_text);
    HBox_configuration_path->addWidget(configuration_path_button);

    /* */
    configuration_params_label = new QLabel("Loaded parameters: ");
    configuration_params_text  = new QTextEdit();
    configuration_params_text->setReadOnly(true);              // 只读显示
    configuration_params_text->setLineWrapMode(QTextEdit::NoWrap);
    configuration_params_text->setMinimumHeight(180);

    QFont monoFont("Courier New");
    monoFont.setStyleHint(QFont::Monospace);
    configuration_params_text->setFont(monoFont);
    

    /* */
    Layout_parameters->addLayout(HBox_configuration_path,0,0,1,2);
    Layout_parameters->addWidget(configuration_params_label, 1, 0, 1, 2);
    Layout_parameters->addWidget(configuration_params_text,  2, 0, 1, 2);

    /* */
    reloadCfgsAndShow();

    /* */
    connect(configuration_path_button, SIGNAL(clicked()),
            this, SLOT(onConfigPathButtonClicked()));

    connect(configuration_path_text, SIGNAL(editingFinished()),
            this, SLOT(onConfigPathEditingFinished()));

    /* */
    groupBox_parameters->setLayout(Layout_parameters);
}

/*-----------------------------------------------------------------------*/
Reconstruction_Dialog ::~Reconstruction_Dialog(){
    cout<<"finish "<<endl;
}

/*-----------------------------------------------------------------------*/
void Reconstruction_Dialog :: Select_input_path(){
    QString select_input_file = QFileDialog::getExistingDirectory(this, "select the TeraFly file.", "/");
    if (select_input_file.isEmpty()) return;
	input_path_value = select_input_file;
	input_path_text->setText(select_input_file);
    cout<<"input_path: "<<input_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: Select_marker_path(){
    QString select_marker_file = QFileDialog::getOpenFileName(this, "select the .marker file.", "", "*.marker *.apo");
    if (select_marker_file.isEmpty()) return;
	marker_path_value = select_marker_file;
	marker_path_text->setText(select_marker_file);
    cout<<"marker_path: "<<marker_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: Select_save_path(){
    QString select_save_file = QFileDialog::getExistingDirectory(this, "select the save file.", "/");
    if (select_save_file.isEmpty()) return;
	save_path_value = select_save_file;
	save_path_text->setText(select_save_file);
    cout<<"save_path: "<<save_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: Select_pytorch_path(){
    QString select_pytorch_file = QFileDialog::getExistingDirectory(this, "select the pytorch file.", "/");
    if (select_pytorch_file.isEmpty()) return;
	pytorch_path_value = select_pytorch_file;
	pytorch_path_text->setText(select_pytorch_file);
    cout<<"pytorch_path: "<<pytorch_path_value.toStdString()<<endl;
}

/*-----------------------------------------------------------------------*/
void Reconstruction_Dialog :: update_save_path(){
    save_path_value = save_path_text->text();
    cout<<"save_path: "<<save_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: update_input_path(){
    input_path_value = input_path_text->text();
    cout<<"input_path: "<<input_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: update_marker_path(){
    marker_path_value = marker_path_text->text();
    cout<<"marker_path: "<<marker_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: update_pytorch_path(){
    pytorch_path_value = pytorch_path_text->text();
    cout<<"pytorch_path: "<<pytorch_path_value.toStdString()<<endl;
}

/*-----------------------------------------------------------------------*/
void Reconstruction_Dialog :: Select_configuration_path(){
    QString select_configuration_file = QFileDialog::getExistingDirectory(this, "select the configuration file.", "/");
    if (select_configuration_file.isEmpty()) return;
	configuration_path_value = select_configuration_file;
	configuration_path_text->setText(select_configuration_file);
    cout<<"configuration_path: "<<configuration_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog :: update_configuration_path(){
    configuration_path_value = configuration_path_text->text();
    cout<<"configuration_path: "<<configuration_path_value.toStdString()<<endl;
}

void Reconstruction_Dialog::reloadCfgsAndShow()
{
    configuration_path_value = configuration_path_text->text();

    cfgs = LoadConfig(configuration_path_value.toStdString());   // 存到成员变量

    QString show;
    std::cout << " ################  cfgs params  ################# " << std::endl;
    for (const auto& kv : cfgs)
    {
        std::cout << kv.first << " : " << kv.second << std::endl;
        show += QString::fromStdString(kv.first) + " : "
              + QString::fromStdString(kv.second) + "\n";
    }
    std::cout << " ############################################## " << std::endl;

    configuration_params_text->setPlainText(show);
}

void Reconstruction_Dialog::onConfigPathButtonClicked()
{
    QString file = QFileDialog::getOpenFileName(this,
                        "Select yaml config file",
                        configuration_path_text->text(),
                        "YAML Files (*.yaml *.yml);;All Files (*)");
    if (file.isEmpty()) return;

    configuration_path_text->setText(file);
    reloadCfgsAndShow();          // 内部会更新 cfgs 成员
}

void Reconstruction_Dialog::onConfigPathEditingFinished()
{
    reloadCfgsAndShow();          // 内部会更新 cfgs 成员
}

/*----------------------------------------------------------------------*/
void Reconstruction_Dialog :: Start_neuron_reconstruction(){
    
    neuron_reconstruction = new Neuron_Reconstruction(_callback);
    neuron_reconstruction->imagePath = input_path_value;
    neuron_reconstruction->markerPath = marker_path_value;
    neuron_reconstruction->outputFolderPath = save_path_value;
    neuron_reconstruction->pytorchPath = pytorch_path_value;

    /* cfgs */
    neuron_reconstruction->Cfgs = cfgs;

    neuron_reconstruction->pythonCodePath = QString::fromStdString(GetCfg(cfgs, "Python_code_path"));;
    neuron_reconstruction->traceMethod = QString::fromStdString(GetCfg(cfgs, "rec_model_name"));
    neuron_reconstruction->segmentMethod = QString::fromStdString(GetCfg(cfgs, "seg_model_name"));
    neuron_reconstruction->blockSize = atoi(GetCfg(cfgs, "blockSize").c_str());
    neuron_reconstruction->margin_lamd = atof(GetCfg(cfgs, "margin_lamd").c_str());;
    neuron_reconstruction->MaxpointNUM = atoi(GetCfg(cfgs, "MaxpointNUM").c_str());
    neuron_reconstruction->min_branch_length = atoi(GetCfg(cfgs, "min_branch_length").c_str());
    
    neuron_reconstruction->SINGLE_NEURON_RCONSTRUCT();
    delete neuron_reconstruction;
    cout<<"finished"<<endl;
}