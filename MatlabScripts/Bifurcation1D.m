a = csvread('C:\Users\KiShiVi\Desktop\mat.csv');
figure(1);
plot(a(:,1), a(:,2), 'r.', 'MarkerSize', 1);

%b = csvread('C:\Users\KiShiVi\Desktop\mat1.csv');

%hold on;
%figure(2);
%plot(b(:,1), b(:,2), 'r.', 'MarkerSize', 1);

% x = linspace(0.05, 0.35, 1000)
% image(x, x, a)

% pcolor(a);
% axis ij;
% axis square;
% grid minor