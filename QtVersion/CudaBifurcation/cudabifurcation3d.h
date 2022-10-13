#pragma once

#include <QWidget>

class QSpinBox;
class QDoubleSpinBox;
class QProgressBar;
class QLineEdit;
class QThread;
class QPushButton;
class QMessageBox;
class QComboBox;

class CudaBifurcation3D : public QWidget
{
    Q_OBJECT

public:
    CudaBifurcation3D(QWidget* parent = nullptr);
    ~CudaBifurcation3D();

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
    void onTimeoutTimer();
    void onStopButtonClicked();

private:
    bool                        isCalculate;
    std::atomic<int>            progress;       // WARNING!!! Обращаться только из слота таймера. Не допускать одновременное обращение! (в рот сосал многопоточность)

    QThread* thread;

    QSpinBox* p_tMax;
    QSpinBox* p_nPts;
    QDoubleSpinBox* p_h;

    QLineEdit* p_initialConditions;

    QDoubleSpinBox* p_paramValues1;
    QDoubleSpinBox* p_paramValues2;
    QDoubleSpinBox* p_paramValues3;
    QDoubleSpinBox* p_paramValues4;
    QDoubleSpinBox* p_paramValues5;
    QDoubleSpinBox* p_paramValues6;

    QSpinBox* p_nValue;

    QDoubleSpinBox* p_prePeakFinder;

    QSpinBox* p_thresholdValueOfMaxSignalValue;

    QComboBox* p_discreteModelMode;

    QLineEdit* p_params;

    QSpinBox* p_mode1;
    QSpinBox* p_mode2;
    QSpinBox* p_mode3;

    QSpinBox* p_kdeSampling;

    QDoubleSpinBox* p_kdeSamplesInterval1;
    QDoubleSpinBox* p_kdeSamplesInterval2;

    QDoubleSpinBox* p_kdeSmooth;

    QDoubleSpinBox* p_memoryLimit;

    QLineEdit* p_filePath;

    QProgressBar* p_progressBar;
    QPushButton* p_applyButton;
    QPushButton* p_stopButton;

    QMessageBox* p_successMsgBox;

    quint64                     timeOfCalculate;
};
