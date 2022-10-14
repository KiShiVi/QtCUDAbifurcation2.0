a = csvread('C:\Users\KiShiVi\Desktop\mat.csv');

x = linspace(a(1,1), a(1,2), 10);
y = linspace(a(2,1), a(2,2), 10);
z = linspace(a(3,1), a(3,2), 10);
a(1,:) = [];
a(1,:) = [];
a(1,:) = [];

arr = zeros(length(a(1,:)) - 1, length(a(1,:)) - 1, length(a(1,:)) - 1);

for i = 1:(length(a(1,:)) - 1)
    arr(i,:,:) = a(1:length(a(1,:)) - 1, 1:length(a(1,:)) - 1);
    a(1:length(a(1,:)) - 1,:) = [];
end