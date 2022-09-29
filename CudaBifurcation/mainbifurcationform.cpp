#include "mainbifurcationform.h"
#include "cudabifurcation.h"
#include "cudabifurcation2d.h"
#include "cudabifurcation3d.h"

#include <QTabWidget>
#include <QVBoxLayout>

MainBifurcationForm::MainBifurcationForm(QWidget* parent)
	: QWidget(parent)
{
	QTabWidget* tabWidget = new QTabWidget();
	CudaBifurcation* bidurcation1DWidget = new CudaBifurcation();
	CudaBifurcation2D* bidurcation2DWidget = new CudaBifurcation2D();
	CudaBifurcation3D* bidurcation3DWidget = new CudaBifurcation3D();

	tabWidget->addTab(bidurcation1DWidget, "1D");
	tabWidget->addTab(bidurcation2DWidget, "2D");
	tabWidget->addTab(bidurcation3DWidget, "3D");

	QVBoxLayout* mainLayout = new QVBoxLayout();
	mainLayout->addWidget(tabWidget);

	setLayout(mainLayout);
}

MainBifurcationForm::~MainBifurcationForm()
{}
