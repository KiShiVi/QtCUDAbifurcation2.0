#include "cudabifurcation.h"
#include "bifurcationKernel.cuh"

#include <QtCore/qdebug.h>
#include <QtWidgets/qspinbox.h>
#include <QtWidgets/qprogressbar.h>
#include <QtWidgets/qlayout.h>
#include <QtWidgets/qlabel.h>
#include <QtWidgets/qlineedit.h>
#include <QtWidgets/qpushbutton.h>
#include <QtCore/qtimer.h>
#include <QtCore/qthread.h>
#include <QtCore/qdatetime.h>
#include <QtWidgets/qmessagebox.h>

#include <fstream>
#include <string>

#define CONF1 "conf1d.txt"

CudaBifurcation::CudaBifurcation(QWidget *parent)
    : QWidget(parent)
{
    initGui();
    parseFile(CONF1);
    QTimer* timer = new QTimer;
    connect(timer, SIGNAL(timeout()), SLOT(onTimeoutTimer()));
    connect(this, SIGNAL(finishBifurcation()), SLOT(onFinishBifurcation()));
    timer->start(10);
    p_progressBar->setMaximum(100);
    isCalculate = false;
}

CudaBifurcation::~CudaBifurcation()
{

}

void CudaBifurcation::callBifurcation()
{
    progress.store(0, std::memory_order_seq_cst);
    isCalculate = true;

    float* params               = new float[p_params->text().split(' ').count()];
    float* initialConditions    = new float[p_initialConditions->text().split(' ').count()];

    for (int i = 0; i < p_params->text().split(' ').count(); ++i)
        params[i] = p_params->text().split(' ')[i].toFloat();

    for (int i = 0; i < p_initialConditions->text().split(' ').count(); ++i)
        initialConditions[i] = p_initialConditions->text().split(' ')[i].toFloat();


    bifurcation1D(  p_tMax->value(),
                    p_nPts->value(),
                    p_h->value(),
                    initialConditions,
                    p_paramValues1->value(),
                    p_paramValues2->value(),
                    p_nValue->value(),
                    p_prePeakFinder->value(),
                    p_thresholdValueOfMaxSignalValue->value(),
                    p_params->text().split(' ').count(),
                    params,
                    p_mode->value(),
                    p_memoryLimit->value(),
                    p_filePath->text().toStdString(),
                    0,
                    progress);

    delete[] params;
    delete[] initialConditions;

    isCalculate = false;
    emit finishBifurcation();
}

void CudaBifurcation::onFinishBifurcation()
{
    progress.store(100, std::memory_order_seq_cst);
    
    p_successMsgBox->setWindowTitle("Process finished!");
    p_successMsgBox->setText("Elapsed time: " + QString::number(QDateTime::currentSecsSinceEpoch() - timeOfCalculate) + " sec.");
    progress.store(100, std::memory_order_seq_cst);
    p_successMsgBox->show();
}

void CudaBifurcation::onTimeoutTimer()
{
    p_progressBar->setValue(progress.load(std::memory_order_seq_cst));
    if (isCalculate)
    {

        p_applyButton->setEnabled(false);
        p_stopButton->setEnabled(true);
    }
    else
    {
        //p_progressBar->reset();
        p_applyButton->setEnabled(true);
        p_stopButton->setEnabled(false);
    }
}

void CudaBifurcation::saveFile(QString filePath)
{
    std::ofstream in;
    in.open(filePath.toStdString());
    if (!in.is_open())
    {
        qDebug() << "Input file open error!";
        in.close();
    }

    in << "T_MAX: " << p_tMax->value() << "\n";
    in << "N_PTS: " << p_nPts->value() << "\n";
    in << "H: " << p_h->value() << "\n";

    in << "\n#Initial conditions\n";
    in << "INITIAL_CONDITIONS:" << p_initialConditions->text().toStdString() << "\n";

    in << "\n#Calculation range\n";
    in << "PARAM_VALUES_1: " << p_paramValues1->value() << "\n";
    in << "PARAM_VALUES_2: " << p_paramValues2->value() << "\n";

    in << "\n#Calculated parameter\n";
    in << "N_VALUE: " << p_nValue->value() << "\n";

    in << "\n#Cutting points before finding peaks (slice of array PRE_PEAKFINDER_SLICE_K * sizeArr:sizeArr)\n";
    in << "PRE_PEAKFINDER_SLICE_K: " << p_prePeakFinder->value() << "\n";

    in << "\n";
    in << "THRESHOLD_VALUE_OF_MAX_SIGNAL: " << p_thresholdValueOfMaxSignalValue->value() << "\n";

    in << "\n#Params (0 is symmetry)\n";
    in << "PARAMS:" << p_params->text().toStdString() << "\n";

    in << "\n";

    in << "MODE: " << p_mode->value() << "\n";

    in << "\n";
    
    in << "MEMORY_LIMIT: " << p_memoryLimit->value() << "\n";

    in << "\n";

    in << "OUTPATH:" << p_filePath->text().toStdString() << "\n";

    in.close();
}

void CudaBifurcation::parseFile(QString filePath)
{

    p_tMax                      ->setValue(parseValueFromFile(filePath, "T_MAX").toInt());                      //!< Время моделирования
    p_nPts                      ->setValue(parseValueFromFile(filePath, "N_PTS").toInt());                      //!< Кол-во точек
    p_h                         ->setValue(parseValueFromFile(filePath, "H").toFloat());                        //!< Шаг интегрирования

    p_initialConditions         ->setText(parseValueFromFile(filePath, "INITIAL_CONDITIONS"));     //!< Начальные условия x

    p_paramValues1              ->setValue(parseValueFromFile(filePath, "PARAM_VALUES_1").toFloat());           //!< Начало диапазона расчета
    p_paramValues2              ->setValue(parseValueFromFile(filePath, "PARAM_VALUES_2").toFloat());           //!< Конец диапазона расчета

    p_nValue                    ->setValue(parseValueFromFile(filePath, "N_VALUE").toInt());                    //!< Какую координату (0/1/2 = x/y/z) берем в расчет
    p_prePeakFinder             ->setValue(parseValueFromFile(filePath, "PRE_PEAKFINDER_SLICE_K").toFloat());   //!< Какой процент точек отрезаем (отсекам переходный процесс)

    p_thresholdValueOfMaxSignalValue ->setValue(parseValueFromFile(filePath, "THRESHOLD_VALUE_OF_MAX_SIGNAL").toInt());

    p_params                    ->setText(parseValueFromFile(filePath, "PARAMS"));

    p_mode                      ->setValue(parseValueFromFile(filePath, "MODE").toInt());                       //!< По какому параметру обходим (см. enum Mode)

    p_memoryLimit               ->setValue(parseValueFromFile(filePath, "MEMORY_LIMIT").toFloat());             //!< По какому параметру обходим (см. enum Mode)

    p_filePath                  ->setText(parseValueFromFile(filePath, "OUTPATH"));                             //!< Путь к файлу с результатом 
}

void CudaBifurcation::initGui()
{
    p_successMsgBox         = new QMessageBox;

    p_tMax                  = new QSpinBox;
    p_nPts                  = new QSpinBox;
    p_h                     = new QDoubleSpinBox;

    p_initialConditions     = new QLineEdit();

    p_paramValues1          = new QDoubleSpinBox;
    p_paramValues2          = new QDoubleSpinBox;

    p_nValue                = new QSpinBox;

    p_prePeakFinder         = new QDoubleSpinBox;

    p_thresholdValueOfMaxSignalValue = new QSpinBox();

    p_params                = new QLineEdit();

    p_mode                  = new QSpinBox;

    p_memoryLimit           = new QDoubleSpinBox;

    p_filePath              = new QLineEdit;

    p_progressBar           = new QProgressBar;

    QHBoxLayout* tMaxLayout = new QHBoxLayout();
    tMaxLayout->addWidget(new QLabel("TMax: "));
    tMaxLayout->addWidget(p_tMax);
    p_tMax->setMinimum(10);
    p_tMax->setMaximum(99999999);
    p_tMax->setValue(100);

    QHBoxLayout* nPtsLayout = new QHBoxLayout();
    nPtsLayout->addWidget(new QLabel("NPts: "));
    nPtsLayout->addWidget(p_nPts);
    p_nPts->setMinimum(10);
    p_nPts->setMaximum(99999999);
    p_nPts->setValue(100);

    QHBoxLayout* hLayout = new QHBoxLayout();
    hLayout->addWidget(new QLabel("H: "));
    hLayout->addWidget(p_h);
    p_h->setDecimals(9);
    p_h->setMinimum(0.000000001);
    p_h->setMaximum(99999999);
    p_h->setValue(0.01);

    QHBoxLayout* initialConditionLayout = new QHBoxLayout();
    initialConditionLayout->addWidget(p_initialConditions);

    QHBoxLayout* paramValuesLayout = new QHBoxLayout();
    paramValuesLayout->addWidget(p_paramValues1);
    paramValuesLayout->addWidget(p_paramValues2);
    p_paramValues1->setMinimum(0);
    p_paramValues1->setMaximum(99999999);
    p_paramValues1->setValue(0);
    p_paramValues2->setMinimum(0);
    p_paramValues2->setMaximum(99999999);
    p_paramValues2->setValue(0);

    QHBoxLayout* nValueLayout = new QHBoxLayout();
    nValueLayout->addWidget(new QLabel("NValue: "));
    nValueLayout->addWidget(p_nValue);
    p_nValue->setMinimum(0);
    p_nValue->setMaximum(2);
    p_nValue->setValue(0);

    QHBoxLayout* prePeakFinderLayout = new QHBoxLayout();
    prePeakFinderLayout->addWidget(new QLabel("Pre peak finder: "));
    prePeakFinderLayout->addWidget(p_prePeakFinder);
    p_prePeakFinder->setMinimum(0);
    p_prePeakFinder->setMaximum(0.95);
    p_prePeakFinder->setValue(0.3);

    QHBoxLayout* p_thresholdValueOfMaxSignalValueLayout = new QHBoxLayout();
    p_thresholdValueOfMaxSignalValueLayout->addWidget(new QLabel("Max Signal Value Threshold: "));
    p_thresholdValueOfMaxSignalValueLayout->addWidget(p_thresholdValueOfMaxSignalValue);
    p_thresholdValueOfMaxSignalValue->setMinimum(0);
    p_thresholdValueOfMaxSignalValue->setMaximum(100000);
    p_thresholdValueOfMaxSignalValue->setValue(10000);

    QHBoxLayout* paramsLayout = new QHBoxLayout();
    paramsLayout->addWidget(p_params);

    QHBoxLayout* modeLayout = new QHBoxLayout();
    modeLayout->addWidget(new QLabel("Mode: "));
    modeLayout->addWidget(p_mode);
    p_mode->setMinimum(0);
    p_mode->setMaximum(3);
    p_mode->setValue(0);

    QHBoxLayout* memoryLimitLayout = new QHBoxLayout();
    memoryLimitLayout->addWidget(new QLabel("Memory limit: "));
    memoryLimitLayout->addWidget(p_memoryLimit);
    p_memoryLimit->setMinimum(0);
    p_memoryLimit->setMaximum(1);
    p_memoryLimit->setValue(0.95);

    QHBoxLayout* filePathLayout = new QHBoxLayout();
    filePathLayout->addWidget(new QLabel("Out file path: "));
    filePathLayout->addWidget(p_filePath);
    p_filePath->setText("conf.txt");

    p_progressBar = new QProgressBar();

    p_applyButton   = new QPushButton("Apply");
    p_stopButton    = new QPushButton("Stop");

    connect(p_applyButton, SIGNAL(clicked()), this, SLOT(onApplyButtonClicked()));
    connect(p_stopButton, SIGNAL(clicked()), this, SLOT(onStopButtonClicked()));

    QVBoxLayout* mainLayout = new QVBoxLayout();
    mainLayout->addLayout(tMaxLayout);
    mainLayout->addLayout(nPtsLayout);
    mainLayout->addLayout(hLayout);
    mainLayout->addWidget(new QLabel("Initial conditions: "));
    mainLayout->addLayout(initialConditionLayout);
    mainLayout->addWidget(new QLabel("Param Values: "));
    mainLayout->addLayout(paramValuesLayout);
    mainLayout->addLayout(nValueLayout);
    mainLayout->addLayout(prePeakFinderLayout);
    mainLayout->addLayout(p_thresholdValueOfMaxSignalValueLayout);
    mainLayout->addWidget(new QLabel("Params: "));
    mainLayout->addLayout(paramsLayout);
    mainLayout->addLayout(modeLayout);
    mainLayout->addLayout(memoryLimitLayout);
    mainLayout->addLayout(filePathLayout);
    mainLayout->addStretch(0);
    mainLayout->addWidget(p_progressBar);
    mainLayout->addWidget(p_applyButton);
    mainLayout->addWidget(p_stopButton);

    setLayout(mainLayout);
}

QString CudaBifurcation::parseValueFromFile(QString filePath, QString parameterName)
{
    std::string inBuffer;
    std::ifstream in;
    in.open(filePath.toStdString());
    if (!in.is_open())
    {
        qDebug() << "Input file open error!";
        return QString();
    }

    while (!in.eof())
    {
        std::getline(in, inBuffer);
        if (inBuffer == "\n")
            continue;
        if (inBuffer.size() > 0 && inBuffer[0] == '#')
            continue;
        if (inBuffer.substr(0, inBuffer.find(":")) == parameterName.toStdString())
            return inBuffer.substr(inBuffer.find(":") + 1, inBuffer.size()).c_str();
    }
    qDebug() << "Input file error! Not found " << parameterName << " parameter!";
    return QString();
}

void CudaBifurcation::onStopButtonClicked()
{
    //QTextCodec::setCodecForCStrings(QTextCodec::codecForName("utf-8"));
    p_successMsgBox->setWindowTitle(":-B");
    p_successMsgBox->setText("Xaxaxa, Hae6a/\\\nKOPO4E IIpo6oBal ocTaHoBuTb IIoToK, Ho GPU He ocTaHaB/\\uBaeT cBou IIpoueccbI :(");
    p_successMsgBox->show();
    //thread->terminate();
    //isCalculate = false;
}

void CudaBifurcation::onApplyButtonClicked()
{
    saveFile(CONF1);
    timeOfCalculate = QDateTime::currentSecsSinceEpoch();
    thread = new QThread();
    moveToThread(thread);
    connect(thread, SIGNAL(started()), this, SLOT(callBifurcation()), Qt::DirectConnection);
    thread->start();
}
