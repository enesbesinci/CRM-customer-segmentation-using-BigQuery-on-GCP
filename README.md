# Customer Segmentation using BigQuery on Workbench on Google Cloud

Hello everyone, in this project, I'll develop a customer segmentation model using BigQuery and other GCP technologies like Google Cloud Storage, Vertex AI and Google Looker Studio.

Customer segmentation is the process of categorizing a business's customers into groups with similar characteristics. These characteristics may include demographic information, shopping habits, or regional location. Through the use of data science tools, this analysis allows businesses to create targeted marketing strategies and personalize services for specific customer groups. This, in turn, can enhance customer satisfaction and provide a competitive advantage for the business.

Let's start.

![Customer-Segmentation](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/671fc106-ab3d-43e4-84b3-596465ba72cc)

## STEP 1: Upload the data to Google Cloud Stroage

First of all we should create a Google Cloud Stroage to upload and store the dataset. There are different kind of storage solutions in GCP but we will use the standart one because of the pricing. And after that we upload the data from our local PC to Google Cloud Stroage. You can find the open-source dataset I used in this project. https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data

![Screenshot 2023-11-30 173507](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/072af758-17d8-4176-8c0f-2714570ab96a)

Our data is now in Google Cloud Stroage, in the next steps we will pull the data from here to BigQuery and Vertex AI Workbench.

## STEP 2: Load the data from Bucket to BigQuery

In this step we pull the data from Bucket to BigQuery. We'll load the data into BigQuery because we'll be pulling data from BigQuery instead of Bucket when we work on the Workbench notebook.

![Screenshot 2023-11-30 173949](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/1eaff5c1-c88d-480f-8d44-b917ba2a33f8)

We loaded the data into BigQuery from Bucket. Now, we can create a K-Means Clustering model using BigQuery if we want. However, in order to show how to use BigQuery and SQL commands on Vertex AI Workbench and because we want to do the data preprocessing steps with python codes, we will be doing all these operations in the Workbench notebook, not on the BigQuery homepage. We will only do the model building phase with BigQuery.

## STEP 3: Instance Creation, Load and Describe the Data from BQ to Workbench Notebook and Data Preprocessing

## A Summary of The Dataset:

This dataset contains data on the demographic and spending habits of a grocery chain's customers.

## Description of Variables:

### Variables About Customers
![Screenshot 2023-11-30 211504](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/30895191-6903-4ce4-b4db-dfb6d6fcfc31)

### Variables About Products and Promotions

![Screenshot 2023-11-30 211519](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/ef36690d-ac7c-4f93-bc74-ee1e6804607d)

### Other Variables

![Screenshot 2023-11-30 211527](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/977acafd-c961-48dc-90bf-e25b91961d5c) 

Yes, now that we have the necessary information about the dataset, we can move on to the next step.

We will use Vertex AI Workbench for the calculations because of the high computing power capacity offered by Google. Now we need to create an Instance to work on. Since our dataset is not very big, we don't need a lot of processing power capacity so an Instance with 2vCPU and 8GB Ram is enough for us.

![Screenshot 2023-11-30 173818](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/7ec1fe8b-82dc-494d-b612-d95478905749)

Yes we created an instance with 2vCPU and 8GB RAM, now let's create a notebook in the instance we created.

![Screenshot 2023-11-30 195414](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/8911003a-b038-42e7-8552-4a04c236af43)

Now we can start writing code.

### Starting to Code...

First of all we need to import the necessary libraries.

![Screenshot 2023-11-30 195639](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/872015b4-dc28-4df6-97d0-1817aedbe606)

Then we create a BigQuery client object to pull the data into the workbench environment, and just like in the BigQuery environment, we pull our data with SQL commands and bring them into a dataframe format. Let's look at the data.

![Screenshot 2023-11-30 195701](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/2e35685a-4c49-47e3-b95a-55d67caead8b)

We have about 2200 row and 29 feature. Let's take a look at our numeric variables.

![Screenshot 2023-11-30 200301](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/f3dea4cc-3729-47c1-a172-259661182e6c)  ![Screenshot 2023-11-30 200332](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/3a89891f-483b-4a2f-8158-78bd06bee657)

At first glance, it seems that there are outliers in some columns of the data set, and the columns "Z_CostContact" and "Z_Revenue" have a standard deviation of 0 (zero), which means that all values in these columns are equal.

Now we write a simple function to see the missing values, this function will return the number of missing values and percentage of missing values.

![Screenshot 2023-11-30 200623](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/3ef42ea8-76be-4fab-8c54-9061748a315c)

Let's see the results.

![Screenshot 2023-11-30 200638](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/52138508-fe29-4121-80f2-564fd164bc5f)  ![Screenshot 2023-11-30 200707](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/45f76857-b30b-41c7-9a75-12703bed2b58)

The 24 missing values in the income column make up only 1.07% of the data. This is a small percentage (not 50%, for example), so we don't need to get rid of the income column. We can replace the missing data with the column's average value. Actually, I won't fill the missing values in the "Income" column with the mean value at this point because we haven't identified and removed any outliers from our dataset. We need to do this first before we can determine whether the values in the "Income" column are symmetrically distributed or not. After we remove any outliers, we will then fill in the missing data with the mean. And on the other hand we will drop the columns "Z_CostContact", "Z_Revenue", because the standard deviation of these two variables is 0, they will not mean anything for our model.


And after that we should check the outliers. There are different methods to identify outliers, in this project I will use the IQR method. Let's see these outliers, I have written a simple function to see outliers, in this function we need to write the values Q1 and Q3.

![Screenshot 2023-11-30 201502](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/1f387523-b1de-4208-82d2-fb85bd4349d2)

As seen above we set the first quartile (Q1) and third quartile (Q3) to 0.10 and 0.90, respectively, to widen the range of outlier values. But still we have outliers, we will drop them.

![Screenshot 2023-11-30 205749](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/7d7a5b22-b903-4c90-9c1a-fd141ad5f211)

After that we save the summary of clean data to a csv file to review it again.

![Screenshot 2023-11-30 215049](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/6c2d0141-f70f-40d5-99c7-d6fca86420a3)

You can see the files you have saved in the "file browser" on the left side.

![Screenshot 2023-11-30 215323](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/98bf720f-2926-4d0d-93f7-5b378af58936)

Now let us look at the distribution of the "Income" column. 

![Screenshot 2023-11-30 215152](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/6534b722-439d-4780-a556-54df64787206)

The result:

![Screenshot 2023-11-30 215618](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/2a701d01-60d8-475e-bf74-2fdf38182325)

The Income column shows a distribution similar to the normal distribution, it looks good.

## Feature Engineering:

First, let's take a look at how many different values each of the variables in the dataset contains.

![Screenshot 2023-11-30 220051](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/3df902aa-e715-4f9b-b89a-2266150118cc)

The results:

![Screenshot 2023-11-30 220110](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/ca0813cf-d89d-45ea-89e0-4fcf92df5a23)

![Screenshot 2023-11-30 220143](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/66e6323b-412e-4eea-8445-0908013440ed)

As you can see above, we have lots of different values in some categorical features, it's too much to encoded, that's why we should handle it. Now we will create some new different features.

![Screenshot 2023-12-05 152929](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/52300bc4-28b1-4e45-8eaf-a9921b0b6735) ![Screenshot 2023-12-05 152947](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/181d3e9f-0e8a-41a3-b556-233c86949e2c)

We have created some new features, let's take a look.

![Screenshot 2023-12-05 153047](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/52eca4f3-d751-420d-9d10-8255fffd95fa)

And after that we save the summary of new clean data to a csv file to review it again and upload the Cloud Storage.

![Screenshot 2023-12-05 153028](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/e198ce1d-8218-4004-9f69-428645e584a8)

## STEP 4: Build a K-Means Clustering Model using BigQuery on Workbench

In this step we will build a clustering model to segment customers. We will build two different segmentation model. The first one will be product-preferences-based and second one will be transaction-based.

First, we copy the dataframe to be used in the K-Means model.

![Screenshot 2023-12-05 153400](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/c49da213-9cbd-4c47-be58-0e8ffd9e1eb2)

Now let's choose our features for the product-preferences-based model

![Screenshot 2023-12-05 153644](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/50f53494-49b0-42c6-b856-d366d5592a36)

Then standardize the product-preferences-based dataset for use in our model

![Screenshot 2023-12-05 154446](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/c4985e59-5398-4226-81bc-5b8c47f2ca07)

Now we can build the K-Means model. In this section, we can either build our model directly on BigQuery (Figure 1) or we can use the same BigQuery codes and create a BigQuery client and build our model on Vertex AI Workbench (Figure 2).

Figure 1:

![Screenshot 2023-12-05 160228](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/4c83934c-950d-4eb3-a973-44b0478b0b6e)

Figure 2:

![Screenshot 2023-12-05 154534](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/13f59b6e-59b2-42fe-b2e5-e08024a39f63)

At this point, you may ask why we divided our dataset into 3 different segments. The reason for this is that I first found the optimum number of clusters by using the Elbow method in the same way as my dataset, so we set our cluster parameter to 3.

![Screenshot 2023-12-05 171436](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/7292080a-4424-437e-ab12-ced2c6fb7412) ![Screenshot 2023-12-05 171453](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/2ab3b912-787e-48de-b17e-2120110fff23)



You can use whatever you want, I will continue on workbench.

Yes, we have created the model. Now let's see the segmentation results for every single customer. For this, we write a BigQuery-SQL commands again and predict the results. You can write commands either on the Workbench (Figure 3) or in the BigQuery interface (Figure 4) to see the results.

Figure 3:

![Screenshot 2023-12-05 154553](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/eb455268-ab8e-4a0d-9839-f86e74f46b60)

Figure 4:

![Screenshot 2023-12-05 161326](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/ca02e140-8bcb-4750-b190-b2bba4ae6fe6)

And save the results to a csv file to review it and import. Then let's look at the top 5 results and how many customers are there in each of these categories?

![Screenshot 2023-12-05 160835](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/1f578db8-87e3-4f86-ba2b-a4a5b237cc62)

As you can see, the centroid_id column is the segmentation result for each customer. But there are numbers for each customers and it doesn't look nice, let's replace the numbers with letters.

![Screenshot 2023-12-05 161608](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/48efe8d9-a9cd-4f9e-8720-4d77c5e64113)

Yes, we have created the model and now we know the segments for every single single customer.

we will now analyze our results to gain a better understanding of them and present them in a clearer format, such as a presentation, to our team or as needed.

Let's see how many customers we have in each category.

![Screenshot 2023-12-05 162426](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/0ee0178d-ab29-40b9-8847-c69e983e8437)

In total, in which category customers preferred which products more.

![Screenshot 2023-12-05 162450](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/2175a948-ebff-45ca-ba00-caf62a51fa8b)

We should remember that the more crowded a cluster is, the higher its total consumption is likely to be, so let's look at which products each segment prefers more on average.

![Screenshot 2023-12-05 162708](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/e7b6dbab-b93a-4c86-a988-ababeb1db2ef)

Finally, let's take a look at the consumption of gold and wine by clients and see how they differ from each other.

![Screenshot 2023-12-05 163004](https://github.com/enesbesinci/customer-segmentation-using-BigQuery-on-GCP/assets/110482608/2ba7b43e-45e8-4478-8657-ae86b37e8fe0)

We can identify the differences between clusters. At this stage, we now know which customers belong to which cluster and are aware of the product preferences within these clusters. From this point on, we should develop specialized sales, advertising, and marketing activities for these segments. For instance, for a segment with high alcohol and meat consumption, we can channel consumption through techniques such as advertisements, coupons, and other promotional strategies tailored to these products. It is essential to consider the characteristics of each segment and design targeted marketing solutions accordingly.


































