#include "cudabifurcation.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    CudaBifurcation w;
    w.show();
    return a.exec();
}
