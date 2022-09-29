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
    void saveFile(QString filePath);
    void parseFile(QString filePath);
    void initGui();
    QString parseValueFromFile(QString filePath, QString parameterName);

private slots:
    void callBifurcation();
    void onFinishBifurcation();
    void onApplyButtonClicked();
    void onStopButtonClicked();
    void onTimeoutTimer();

private:
    bool                        isCalculate;
    std::atomic<int>            progress;       // WARNING!!! Обращаться только из слота таймера. Не допускать одновременное обращение! (в рот сосал многопоточность)

    QThread*                    thread;

    QSpinBox*                   p_tMax;
    QSpinBox*                   p_nPts;
    QDoubleSpinBox*             p_h;

    QLineEdit*                  p_initialConditions;

    QDoubleSpinBox*             p_paramValues1;
    QDoubleSpinBox*             p_paramValues2;

    QSpinBox*                   p_nValue;

    QDoubleSpinBox*             p_prePeakFinder;

    QSpinBox*                   p_thresholdValueOfMaxSignalValue;

    QLineEdit*                  p_params;

    QSpinBox*                   p_mode;

    QDoubleSpinBox*             p_memoryLimit;

    QLineEdit*                  p_filePath;

    QProgressBar*               p_progressBar;
    QPushButton*                p_applyButton;
    QPushButton*                p_stopButton;

    QMessageBox*                p_successMsgBox;

    quint64                     timeOfCalculate;
};
