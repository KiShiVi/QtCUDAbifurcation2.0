#pragma once

#include <QtWidgets/qwidget.h>

class QSpinBox;
class QDoubleSpinBox;
class QProgressBar;
class QLineEdit;
class QThread;
class QPushButton;
class QMessageBox;

class CudaBifurcation : public QWidget
{
    Q_OBJECT

public:
    CudaBifurcation(QWidget *parent = nullptr);
    ~CudaBifurcation();

signals:
        void finishBifurcation();

private:
    void parseFile(QString filePath);
    void initGui();
    QString parseValueFromFile(QString filePath, QString parameterName);

private slots:
    void callBifurcation();
    void onFinishBifurcation();
    void onApplyButtonClicked();
    void onTimeoutTimer();

private:
    bool                        isCalculate;
    std::atomic<int>            progress;       // WARNING!!! Обращаться только из слота таймера. Не допускать одновременное обращение! (в рот сосал многопоточность)

    QThread*                    thread;

    QSpinBox*                   p_tMax;
    QSpinBox*                   p_nPts;
    QDoubleSpinBox*             p_h;

    QDoubleSpinBox*             p_initialCondition1;
    QDoubleSpinBox*             p_initialCondition2;
    QDoubleSpinBox*             p_initialCondition3;

    QDoubleSpinBox*             p_paramValues1;
    QDoubleSpinBox*             p_paramValues2;

    QSpinBox*                   p_nValue;

    QDoubleSpinBox*             p_prePeakFinder;

    QDoubleSpinBox*             p_paramA;
    QDoubleSpinBox*             p_paramB;
    QDoubleSpinBox*             p_paramC;

    QSpinBox*                   p_mode;

    QDoubleSpinBox*             p_memoryLimit;

    QLineEdit*                  p_filePath;

    QProgressBar*               p_progressBar;
    QPushButton*                p_applyButton;

    QMessageBox*                p_successMsgBox;

    quint64                     timeOfCalculate;
};
