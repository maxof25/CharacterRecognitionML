%%KNN and Decision Tree Algorithms for Character Recognition of EMNIST Dataset

clear all
close all
clc
%Lines 7, 8 and 9 load the EMNIST dataset into the variable 'tab', and each
%column of the data set is stored in the corresponding variables called 
% 'images' and 'letters' for later use.
tab = load('dataset-letters.mat');
images = tab.dataset.images;
letters = tab.dataset.key(tab.dataset.labels());

%Line 13 converts the existing 1*784 Vector into a 28*28 pixel image ready
%to be displayed in figure 1.
double_image = double(reshape(images.', 28, 28, []));

%Line 18 stores the vector stored in the variable 'images' in a table
%called tab2. The next line measures its height to establish the number of
%letters in the ENMIST dataset for later use. This should be 26000.
tab2 = table(images);
row_height = height(tab2);

%The below for loop displays 12 random images and their corresponding lable
%to the window titled figure(1). Each is presented in grayscale.
figure(1)
for h=1 :12
    corr = randi([1 row_height]);
    subplot(3,4,h), imagesc(double_image(:,:,corr)), title(letters(corr)), axis off, colormap gray
end

%The below for loop extends the table called 'tab2' by creating a second
%column called 'letter'. Each of the corresponding letters to the intensity
%data stored in the image column is stored in this column, presenting all
%of our EMNIST data with its correct label.
for i=1: height(tab2)
    j = cellstr(letters(i));
    tab2.letter(i) = categorical(j);
end

%Rows 44 to 52 create 26000 random indices for assigning the data that will
%be used to test and train our machine learning algorithms. our
%training_images, training_letters contain 13000 instances for which we
%will use to train each algorithm and the correspodning testing_images and
%testing_letters will be used to test whether the predicted data is
%correct.
indices = randperm(row_height);
training_set=tab2(indices(1:(row_height/2)),:);
testing_set=tab2(indices(((row_height/2)+1):end),:);

training_images = training_set.images;
training_letters = categorical(training_set.letter);

testing_images = testing_set.images;
testing_letters = categorical(testing_set.letter);

%The two lines below initialise the variable that will be used to store
%each distance metric's predicted lable for use after it has been fed the
%training data.
euclidean_predicted_letters = categorical.empty(size(testing_letters,1),0);
manhattan_predicted_letters = categorical.empty(size(testing_letters,1),0);

times = zeros(2,1);

k=1;

tic;
%The below for loop goes through the testing data to collect the distance
%information and determine the predicted label for a K-Nearest neighbour 
% algorithm using the Euclidian distance metric. 'tic' and 'toc' store the
% time taken for the training and testing to occur, for which are stored in
% the array called 'times'.
for l=1:size(testing_letters,1)
    comp1 = training_images;
    comp2 = repmat(testing_images(l,:),[size(training_images,1),1]);
    distances = sqrt(sum((comp1-comp2).^2,2));
    [~,indices] = sort(distances);
    k_indices = indices(1:k);
    knn = training_letters(k_indices);
    euclidean_predicted_letters(l)=mode(knn);
end
times(1)=toc;

tic;
%Similar to the previous for loop, this for loop goes below through the 
%testing data to collect the distance information and determine the predicted
%label for the KNN algorithm using a Manhatten distance metric. 'Tic' and 'Toc'
%are used to store the time taken to train and test the algorithm in the
%second column of the 'times' variable.
for m=1:size(testing_letters,1)
    comp1 = training_images;
    comp2 = repmat(testing_images(m,:),[size(training_images,1),1]);
    distances = sum(abs(comp1-comp2),2);
    [~,indices] = sort(distances);
    k_indices = indices(1:k);
    knn = training_letters(k_indices);
    manhattan_predicted_letters(m)=mode(knn);
end
times(2)=toc;

%Lines 102 to 108 generate the amound of correct predictions from each of
%the KNN algorithms using the euclidean and manhattan distance metric. The
%accuracy is calculated along with the total time taken and is printed to
%the terminal as a percentage (for the accuracy) and a time in seconds.
euclidean_correct_predictions=sum(euclidean_predicted_letters(:)==testing_letters); 
euclidean_accuracy = euclidean_correct_predictions/size(testing_letters,1);
manhattan_accuracy=sum(manhattan_predicted_letters(:)==testing_letters) /size(testing_letters,1);
fprintf('Accuracy of Euclidean distance: %.2f%%\n', (euclidean_accuracy*100));
fprintf('Time taken to test and train Euclidean distance metric: %.4f seconds \n', times(1));
fprintf('Accuracy of Manhattan distance: %.2f%%\n', (manhattan_accuracy*100));
fprintf('Time taken to test and train Manhattan distance metric: %.4f seconds \n', times(2));

%Lines 116 to 127 construct a confusion matrix for each of the KNN
%algorithms. Each unique letter is displayed on the x and y axis with the
%testing letters on the x axis and predicted on the y. Digits are
%presented on each matrix to explain the number of times the algorithm
%predicted on corresponding letter to be another corresponding letter. This
%helps visualise how arrucate each of the algorithms is.
euclideanCM=confusionmat(testing_letters,euclidean_predicted_letters);
manhattanCM=confusionmat(testing_letters,manhattan_predicted_letters);

figure(2)

subplot(1,2,1);
confusionchart(euclideanCM,unique(testing_letters));
title('Euclidean Confusion Matrix');

subplot(1,2,2);
confusionchart(manhattanCM,unique(testing_letters));
title('Manhattan Confusion Matrix');

times2 = zeros(2,1);

tic;
%Lines 136 to 144 use the pre-existing MATLAB K-Nearest Neighbour Algorithm
%to re run the test above using the same training and testing software. It
%also presents the accuracy and time in an identical fashion to the
%previous tests to the terminal for comparison.
knnmodel = fitcknn(training_images,training_letters);
fitkcnn_predicted = predict(knnmodel, testing_images);
times2(1)=toc;

fitcknn_correct_predictions =sum(fitkcnn_predicted==testing_letters); 
fitcknn_accuracy = fitcknn_correct_predictions/size(testing_letters,1);
fprintf('Accuracy of MATLABs KNN algorithm: %.2f%%\n', (fitcknn_accuracy*100));
fprintf('Time taken to train and test MATLABs K-Nearest Neighbour model: %.4f seconds \n', times2(1));
fitcknn_confusion=confusionmat(testing_letters,fitkcnn_predicted);

tic;
%Lines 151 to 159 use the pre-existing MATLAB Decision Tree Algorithm
%to re-run the test above using the same training and testing software. It
%also presents the accuracy and time in an identical fashion to the
%previous tests to the terminal for comparison.
tree_classifier = fitctree(training_images,training_letters);
tree_predicted = predict(tree_classifier, testing_images);
times2(2)=toc;

tree_correct_predictions =sum(tree_predicted==testing_letters); 
tree_accuracy = tree_correct_predictions/size(testing_letters,1);
fprintf('Accuracy of MATLABs Decsion Tree algorithm: %.2f%%\n', (tree_accuracy*100));
fprintf('Time taken to train and test existing Decision Tree model: %.4f seconds \n', times2(2));
tree_confusion=confusionmat(testing_letters,tree_predicted);

figure(3)

%Lines 167 to 172 display two confusion matrixes to figure 3 in an
%identical fashion the previous confusion matrixes for comprehensive
%comparison of accuracy.
subplot(1,2,1);
confusionchart(fitcknn_confusion,unique(testing_letters));
title('K-Nearest Neighbour Confusion Matrix');

subplot(1,2,2);
confusionchart(tree_confusion,unique(testing_letters));
title('Decision Tree Confusion Matrix');