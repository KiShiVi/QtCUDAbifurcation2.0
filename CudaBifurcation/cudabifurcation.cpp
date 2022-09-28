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

    bifurcation1D(p_tMax->value(),
        p_nPts->value(),
        p_h->value(),
        p_initialCondition1->value(),
        p_initialCondition2->value(),
        p_initialCondition3->value(),
        p_paramValues1->value(),
        p_paramValues2->value(),
        p_nValue->value(),
        p_prePeakFinder->value(),
        p_paramA->value(),
        p_paramB->value(),
        p_paramC->value(),
        p_mode->value(),
        p_memoryLimit->value(),
        p_filePath->text().toStdString(),
        0,
        progress);

    isCalculate = false;
    emit finishBifurcation();
}

void CudaBifurcation::onFinishBifurcation()
{
    progress.store(100, std::memory_order_seq_cst);
    
    p_successMsgBox->setWindowTitle("Success!");
    p_successMsgBox->setText("Bifurcation calculated: " + QString::number(QDateTime::currentSecsSinceEpoch() - timeOfCalculate) + " sec.");
    progress.store(100, std::memory_order_seq_cst);
    p_successMsgBox->show();
}

void CudaBifurcation::onTimeoutTimer()
{
    p_progressBar->setValue(progress.load(std::memory_order_seq_cst));
    if (isCalculate)
    {

        p_applyButton->setEnabled(false);
    }
    else
    {
        //p_progressBar->reset();
        p_applyButton->setEnabled(true);
    }
}

void CudaBifurcation::parseFile(QString filePath)
{

    p_tMax                      ->setValue(parseValueFromFile(filePath, "T_MAX").toInt());                      //!< Время моделирования
    p_nPts                      ->setValue(parseValueFromFile(filePath, "N_PTS").toInt());                      //!< Кол-во точек
    p_h                         ->setValue(parseValueFromFile(filePath, "H").toFloat());                        //!< Шаг интегрирования

    p_initialCondition1         ->setValue(parseValueFromFile(filePath, "INITIAL_CONDITIONS_1").toFloat());     //!< Начальные условия x
    p_initialCondition2         ->setValue(parseValueFromFile(filePath, "INITIAL_CONDITIONS_2").toFloat());     //!< Начальные условия y
    p_initialCondition3         ->setValue(parseValueFromFile(filePath, "INITIAL_CONDITIONS_3").toFloat());     //!< Начальные условия z

    p_paramValues1              ->setValue(parseValueFromFile(filePath, "PARAM_VALUES_1").toFloat());           //!< Начало диапазона расчета
    p_paramValues2              ->setValue(parseValueFromFile(filePath, "PARAM_VALUES_2").toFloat());           //!< Конец диапазона расчета

    p_nValue                    ->setValue(parseValueFromFile(filePath, "N_VALUE").toInt());                    //!< Какую координату (0/1/2 = x/y/z) берем в расчет
    p_prePeakFinder             ->setValue(parseValueFromFile(filePath, "PRE_PEAKFINDER_SLICE_K").toFloat());   //!< Какой процент точек отрезаем (отсекам переходный процесс)

    p_paramA                    ->setValue(parseValueFromFile(filePath, "PARAM_A").toFloat());                  //!< Параметр A
    p_paramB                    ->setValue(parseValueFromFile(filePath, "PARAM_B").toFloat());                  //!< Параметр B
    p_paramC                    ->setValue(parseValueFromFile(filePath, "PARAM_C").toFloat());                  //!< Параметр C

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

    p_initialCondition1     = new QDoubleSpinBox;
    p_initialCondition2     = new QDoubleSpinBox;
    p_initialCondition3     = new QDoubleSpinBox;

    p_paramValues1          = new QDoubleSpinBox;
    p_paramValues2          = new QDoubleSpinBox;

    p_nValue                = new QSpinBox;

    p_prePeakFinder         = new QDoubleSpinBox;

    p_paramA                = new QDoubleSpinBox;
    p_paramB                = new QDoubleSpinBox;
    p_paramC                = new QDoubleSpinBox;

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
    initialConditionLayout->addWidget(p_initialCondition1);
    initialConditionLayout->addWidget(p_initialCondition2);
    initialConditionLayout->addWidget(p_initialCondition3);
    p_initialCondition1->setMinimum(0);
    p_initialCondition1->setMaximum(99999999);
    p_initialCondition1->setValue(0);
    p_initialCondition2->setMinimum(0);
    p_initialCondition2->setMaximum(99999999);
    p_initialCondition2->setValue(0);
    p_initialCondition3->setMinimum(0);
    p_initialCondition3->setMaximum(99999999);
    p_initialCondition3->setValue(0);

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

    QHBoxLayout* paramsLayout = new QHBoxLayout();
    paramsLayout->addWidget(p_paramA);
    paramsLayout->addWidget(p_paramB);
    paramsLayout->addWidget(p_paramC);

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

    p_applyButton = new QPushButton("Apply");
    connect(p_applyButton, SIGNAL(clicked()), this, SLOT(onApplyButtonClicked()));

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
    mainLayout->addWidget(new QLabel("Params: "));
    mainLayout->addLayout(paramsLayout);
    mainLayout->addLayout(modeLayout);
    mainLayout->addLayout(memoryLimitLayout);
    mainLayout->addLayout(filePathLayout);
    mainLayout->addStretch(0);
    mainLayout->addWidget(p_progressBar);
    mainLayout->addWidget(p_applyButton);

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

void CudaBifurcation::onApplyButtonClicked()
{
    timeOfCalculate = QDateTime::currentSecsSinceEpoch();
    thread = new QThread();
    moveToThread(thread);
    connect(thread, SIGNAL(started()), this, SLOT(callBifurcation()), Qt::DirectConnection);
    thread->start();
}
