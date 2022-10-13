#include "mainbifurcationform.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainBifurcationForm* w = new MainBifurcationForm();
    w->show();
    return a.exec();
}
