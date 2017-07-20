% Machine Learning @ NCTU EE
% 0310128 Daniel You
% Homework 1

% Read training data from the given file
RawData = csvread('X_test.csv');
% First column for x coordinate, second for y coordinate
x_cord = RawData(: , 1:1);
y_cord = RawData(: , 2:2);
z_cord = csvread('ML.csv');
%z_cord = csvread('MAP.csv');
%z_cord = csvread('Bayesian.csv');


x = reshape(x_cord, 100, 100);
y = reshape(y_cord, 100, 100);
z = reshape(z_cord, 100, 100);
figure
contour(x, y, z);
title('ML (normal)')
%title('MAP (normal)')
%title('Bayesian (normal)')
xlabel('x');
ylabel('y');
