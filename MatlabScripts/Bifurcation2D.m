a = csvread('C:\Users\KiShiVi\Desktop\mat.csv');

x = linspace(a(1,1), a(1,2), 10);
y = linspace(a(2,1), a(2,2), 10);
a(1,:) = [];
a(1,:) = [];

figure(1);
image(x, y, a);
caxis([0, 50]);


%b = csvread('C:\Users\KiShiVi\Desktop\mat1.csv');

%x1 = linspace(b(1,1), b(1,2), 10);
%y1 = linspace(b(2,1), b(2,2), 10);
%b(1,:) = [];
%b(1,:) = [];

%figure(2);
%image(x1, y1, b);
%caxis([0, 50]);

% pcolor(a);
% axis ij;
%axis square;
% grid minor