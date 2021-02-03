% Principle Component Analysis 
% Recognition with Eigenfaces on ORL dataset

% 0.Download ORL dataset, resize images
% 112x92 -> 56x46 to save computation time
dir = 's';
train_imgs = zeros(56,46,200);
test_imgs = zeros(56,46,200); 
train_index = 1;
test_index = 1;
disp('Loading in images...')
for i = 1:40
    subject_dir = strcat(dir, int2str(i));
    rand_list = randperm(10);
    for j = rand_list(1:5)
       img_path = strcat(subject_dir, '/', int2str(j), '.pgm');
       img = imread(img_path);
       img = imresize(img, [56 46]);
       train_imgs(:,:,train_index) = img;
       train_index = train_index + 1;
    end
    for j = rand_list(6:10)
       img_path = strcat(subject_dir, '/', int2str(j), '.pgm');
       img = imread(img_path);
       img = imresize(img, [56 46]);
       test_imgs(:,:,test_index) = img;
       test_index = test_index + 1;
    end
end
disp('Images loaded!')

% 1.Convert each image to a column vector
train_imgs = reshape(train_imgs, [2576,200]);
test_imgs = reshape(test_imgs, [2576,200]);

% 2.Calculate the covariance matrix of all
% training images
imgs_mean = mean(train_imgs, 2);
%imshow(reshape(imgs_mean, [56,46]), [ ])
img_diff = train_imgs - imgs_mean;
C = (1/200).*img_diff*img_diff';
% Run PCA, finding the eigenfaces
[V, D] = eig(C);
% Choose 400 eigenfaces containing most info
eigenfaces = V(:, 1:400);
%eigenface = reshape(eigenfaces(:,1), [56,46]);
%imshow(eigenface, [])

% 3.Calculate projections along each new
% PC axis (i.e. coefficients of each eigenface)
% for each training image
alpha = train_imgs'*eigenfaces;
% Generate the single training image from the
% eigenfaces and display
train_index = 1;
train_img = train_imgs(:,train_index);
%imshow(reshape(train_img, [56,46]), [ ]);
train_alpha = train_img'*eigenfaces;
alpha_img = imgs_mean + eigenfaces*train_alpha';
%alpha_img = reshape(alpha_img, [56,46]);
%imshow(alpha_img, [ ])

% 4.Select a probe/test image and project
% along PC axes (i.e. construct from eigenfaces)
test_index = 1;
test_img = test_imgs(:,test_index);
test_alpha = test_img'*eigenfaces;
% Generate the test image from the eigenfaces and
% display
test_alpha_img = imgs_mean + eigenfaces*test_alpha';
test_alpha_img = reshape(test_alpha_img, [56,46]);
imshow(test_alpha_img, [ ])