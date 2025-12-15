#### Important Library 
library(tidyverse)
library(data.table)
library(lme4)
library(plyr)
library(doFuture)
library(doMC)
library(plyr)
library(dplyr)
library(AGHmatrix)
library(EMMREML)

##Load in all required TRAIN_data for training

Pheno_data_2014_23 <- fread(file = "/usr/users/osatohanmwen/G2F_competition_2024/Training_data/1_Training_Trait_Data_2014_2023.csv")
Meta_data_2014_23 <- fread(file = "/usr/users/osatohanmwen/G2F_competition_2024/Training_data/2_Training_Meta_Data_2014_2023.csv")
Weather_data_2014_23_year <- fread("/usr/users/osatohanmwen/G2F_competition_2024/Training_data/4_Training_Weather_Data_2014_2023_full_year.csv")
Weather_data_2014_23_season <- fread("/usr/users/osatohanmwen/G2F_competition_2024/Training_data/4_Training_Weather_Data_2014_2023_seasons_only.csv")
EC_data_2014_21 <- fread("/usr/users/osatohanmwen/G2F_competition_2024/Training_data/6_Training_EC_Data_2014_2023.csv")
Soil_data_2015_23 <- fread("/usr/users/osatohanmwen/G2F_competition_2024/Training_data/3_Training_Soil_Data_2015_2023.csv")
Key_inbreds <- fread("/usr/users/osatohanmwen/G2F_competition_2024/Training_data/key_inbreds_G2F_2014-2025.txt")
SNP_data_Vcf <- tibble(fread(file = "/usr/users/osatohanmwen/G2F_competition_2024/Training_data/5_Genotype_Data_All_2014_2025_Hybrids.vcf"))


#Load in all required Test_data for training

###Load in all required Test_data for training
Test_data <- fread(file = "/usr/users/osatohanmwen/G2F_competition_2024/Testing_data/1_Submission_Template_2024.csv")
Test_data_meta <- fread(file = "/usr/users/osatohanmwen/G2F_competition_2024/Testing_data/2_Testing_Meta_Data_2024.csv")
Test_data_soil <- fread(file = "/usr/users/osatohanmwen/G2F_competition_2024/Testing_data/3_Testing_Soil_Data_2024.csv")
Test_data_EC <- fread(file = "/usr/users/osatohanmwen/G2F_competition_2024/Testing_data/6_Testing_EC_Data_2024.csv")
Test_data_weather_year  <- fread(file = "/usr/users/osatohanmwen/G2F_competition_2024/Testing_data/4_Testing_Weather_Data_2024_full_year.csv")
Test_data_weather_season  <- fread(file = "/usr/users/osatohanmwen/G2F_competition_2024/Testing_data/4_Testing_Weather_Data_2024_seasons_only.csv")


Test_data <- Test_data %>%
  separate(Env, c("Field_Location", "Year"),"_", remove=FALSE)%>%
  mutate(Hybrid_Field = paste0(Hybrid,"_",Field_Location))%>%
  select(c("Hybrid","Field_Location","Year","Env","Hybrid_Field","Yield_Mg_ha"))


test_hybrid <- unique(Test_data$Hybrid)
test_environment <-  unique(Test_data$Field_Location)

### Wangle the train data

Pheno_data_env <- Pheno_data_2014_23%>%
  mutate(Hybrid_Env = paste0(Hybrid, "|",Env))%>%
  select(c("Hybrid_Env","Hybrid","Env","Year","Field_Location","Experiment","Replicate","Block","Yield_Mg_ha")) %>%
  filter(Year %in% c("2016","2017"))

##Removing missing phenotype

na.idex <- which(is.na(Pheno_data_env$Yield_Mg_ha))
Pheno_data <- Pheno_data_env[-na.idex,]

# set.seed(200)
# sample_rand <- sample(nrow(Pheno_data_env), 1000, replace = FALSE)
# Pheno_data <- Pheno_data_env[sample_rand,]
# str(Pheno_data)


#Identify and Remove outliers using residuals

#### Preparing for modeling ####
Pheno_data$Year <- as.factor(Pheno_data$Year)
Pheno_data$Hybrid<- as.factor(Pheno_data$Hybrid)
Pheno_data$Replicate <- as.factor(Pheno_data$Replicate)
Pheno_data$Block <- as.factor(Pheno_data$Block)
Pheno_data$Field_Location <- as.factor(Pheno_data$Field_Location)
Pheno_data$Hybrid_Env <- as.factor(Pheno_data$Hybrid_Env)
Pheno_data$Yield_Mg_ha <- as.numeric(Pheno_data$Yield_Mg_ha )

#Fitting a linear model with hybrid and replicate as fixed effects in unique environment. The environment here is the field location and year.

### a function that finds outliers using the resiuals from a linear model and remove the outliers.

outlier.resid <- function(data, trait) {
  # Filter rows with complete cases for the given trait
  data <- data[complete.cases(data[[trait]]), ]

  # Fit a linear model for the residuals
  Model <- lm(as.formula(paste0(trait, "~ Hybrid_Env + Replicate")), data = data)

  # Add residuals to the data
  data$RESID <- resid(Model)

  # Calculate two standard deviations of the residuals
  SD2 <- 2 * sd(data$RESID)

  # Flag outliers as 1 if residual exceeds 2 SD
  data$out_flag <- ifelse(abs(data$RESID) > SD2, 1, 0)

  # Save the data with outliers marked
  write.table(data, file = 'Pheno_data_Outlier.csv', col.names = TRUE, row.names = FALSE, sep = '\t', quote = FALSE)

  # Filter out the outliers
  out_data <- data[data$out_flag == 0, ]

  # Save the filtered data
  write.table(out_data, file = 'Pheno_data_without_Outlier.csv', col.names = TRUE, row.names = FALSE, sep = '\t', quote = FALSE)

  # Return the filtered data
  return(out_data)
}

Pheno_data_without_outlier <-outlier.resid(Pheno_data, trait = "Yield_Mg_ha")


Pheno_data <- rbind(Pheno_data_without_outlier,X2014_2015Pheno_data_without_Outlier,X2022_2023Pheno_data_without_Outlier)



BLUES_All_Year=list()

traits <- "Yield_Mg_ha"

Years = c("2014","2015","2016","2017","2022","2023")

j=1

for (k in 1:length(Years)) {

  print(paste0(' Year:', Years[k]))

  data = filter(Pheno_data, Year %in% Years[k])

  for (i in unique(data$Field_Location)) {

    if (n_distinct(data[data$Field_Location == i & !is.na(data$Field_Location), 'Replicate']) > 1){

      print(paste0(i,' : at least 2 reps'))
      mod1 = lm(paste0(traits, " ~ 0 + Hybrid + Replicate "), data=data[data$Field_Location == i &
                                                                          !is.na(data$Field_Location),])
      mean_df = data.frame(Years[k], i,head(gsub("Hybrid", "", row.names(as.data.frame(coef(mod1)))),-1),
                           head(as.data.frame(coef(mod1))[,1], -1),paste0(i, "_",Years[k]))

      colnames(mean_df) =c("Year", "Field_Location", "Hybrid","Yield_Mg_ha", "Env" )


      BLUES_All_Year[[j]] = cbind(mean_df)
      j = 1 + j
    }

    else{
      print(paste0(i,' : only 1 rep'))

      BLUES_All_Year[[j]] = data[data$Field_Location == i &!is.na(data$Yield_Mg_ha),
                                 c("Year", "Field_Location", "Hybrid","Yield_Mg_ha","Env")]
    }
    j = 1 + j
  }
}

BLUES_All=ldply(BLUES_All_Year, rbind)



Pheno_data_final <- BLUES_All %>%
  mutate(Hybrid_Env=paste0(Hybrid, "_",Env)) %>%
  select("Hybrid" ,"Field_Location","Year","Env","Hybrid_Env","Yield_Mg_ha")

BLUES_All_18_21$Field_Location <- BLUES_All_18_21$Field.Location
BLUES_All_18_21$Yield_Mg_ha <- BLUES_All_18_21$Yield.Mg.ha

BLUES_All_18_21 <- BLUES_All_18_21 %>%
  mutate(Hybrid_Env=paste0(Hybrid, "_",Env)) %>%
  select("Hybrid" ,"Field_Location","Year","Env","Hybrid_Env","Yield_Mg_ha")

Pheno_data_final <- rbind(Pheno_data_final, BLUES_All_18_21)

# Remove rows with any negative values
Pheno_data_final <- Pheno_data_final %>%
  filter(!grepl("Replicate", Hybrid))%>%
  filter(Field_Location %in% test_environment)


# #%>%filter(!Year %in% c("2014", "2015"))
write.table(Pheno_data_final,file = 'Phenotype_Blue.csv',col.names = T,row.names = F,sep = '\t',quote = F)

Pheno_data_final <- data.frame(fread(file = "/usr/users/osatohanmwen/G2F_competition_2024/Phenotype_Blue.csv"))%>%
  filter(Year %in% c("2019", "2020","2021", "2022","2023"))

#for Weather data

##Aggregating the environmental data for the training set

###taking the mean of the weather data

Weather_data_2014_23_year <- Weather_data_2014_23_year %>%
  mutate(Env = case_when(
    Env== "MOH1_1_2018" ~ "MOH1_2018",
    Env== "MOH1_1_2020" ~ "MOH1_2020",
    Env== "MOH1_2_2018" ~ "MOH1_2018",
    Env== "MOH1_2_2020" ~ "MOH1_2020",
    TRUE ~ Env
  ))

Weather_data <- aggregate(Weather_data_2014_23_year[,3:18], by=list(Weather_data_2014_23_year$Env), FUN=mean)
colnames(Weather_data)[1] <- "Env"

PhenoWeather_data_final <- Pheno_data_final %>%
  left_join(., Weather_data, by="Env")

summary(PhenoWeather_data_final)

write.table(PhenoWeather_data_final,file = 'PhenoWeather_data_final.csv',col.names = T,row.names = F,sep = '\t',quote = F)

##for Ec data
EC_data_2014_21 <- EC_data_2014_21 %>%
  mutate(Env = case_when(
    Env== "MOH1_1_2018" ~ "MOH1_2018",
    Env== "MOH1_1_2020" ~ "MOH1_2020",
    Env== "MOH1_2_2018" ~ "MOH1_2018",
    Env== "MOH1_2_2020" ~ "MOH1_2020",
    TRUE ~ Env
  )) %>%
  distinct(Env, .keep_all = TRUE)

PhenoEC_data_final <- Pheno_data_final %>%
  left_join(., EC_data_2014_21, by="Env") %>%
  filter(rowMeans(is.na(.)) < 0.5)

summary(PhenoEC_data_final$CumHI30_pFloFla)


write.table(PhenoEC_data_final,file = 'PhenoEC_data_final.csv',col.names = T,row.names = F,sep = '\t',quote = F)

###for meta data
Meta_data <- Meta_data_2014_23 %>%
  separate(Env, c("Field_Location", "Year"),"_", remove=FALSE)%>%
  filter(Field_Location %in% test_environment)%>%
  select(c("Env","Treatment","City","Previous_Crop","Irrigated","Latitude_of_Field_Corner_#1 (lower left)",
           "Longitude_of_Field_Corner_#1 (lower left)","Latitude_of_Field_Corner_#2 (lower right)",
           "Longitude_of_Field_Corner_#2 (lower right)","Latitude_of_Field_Corner_#3 (upper right)",
           "Longitude_of_Field_Corner_#3 (upper right)","Latitude_of_Field_Corner_#4 (upper left)",
           "Longitude_of_Field_Corner_#4 (upper left)","Weather_Station_Latitude (in decimal numbers NOT DMS)",
           "Weather_Station_Longitude (in decimal numbers NOT DMS)"))

Metal_others=Meta_data %>% select(c("Env","Treatment","City","Previous_Crop","Irrigated"))

Metal_final <- Meta_data %>%
  select(-c(Treatment, City, Previous_Crop, Irrigated))

PhenoMetal_data_final <- Pheno_data_final %>%
  left_join(., Metal_final, by="Env")%>%
  filter(rowMeans(is.na(.)) < 0.5)

summary(PhenoMetal_data_final)

# Replace NA values with group mean using ave

for (col in names(PhenoMetal_data_final)) {
  if (is.numeric(PhenoMetal_data_final[[col]])) {  # Ensure it's a numeric column
    for (loc in unique(PhenoMetal_data_final$Field_Location)) {
      # Calculate the mean of the column for the specific Field_Location, excluding NAs
      group_mean <- mean(PhenoMetal_data_final[PhenoMetal_data_final$Field_Location == loc, col], na.rm = TRUE)

      # Replace NA values with the calculated mean within that group
      PhenoMetal_data_final[PhenoMetal_data_final$Field_Location == loc & is.na(PhenoMetal_data_final[[col]]), col] <- group_mean
    }
  }
}

write.table(PhenoMetal_data_final,file = 'PhenoMetal_data_final.csv',col.names = T,row.names = F,sep = '\t',quote = F)

###for soil data
Soil_data<- Soil_data_2015_23 %>%
  separate(Env, c("Field_Location", "Year"),"_", remove=FALSE)

Soil_data <- Soil_data[,-c(2:6,30:37)]

PhenoSoil_data_final <- Pheno_data_final %>%
  left_join(., Soil_data, by="Env")%>%
  filter(!Year %in% c("2014","2015")) %>%
  filter(rowMeans(is.na(.)) < 0.5)

summary(PhenoSoil_data_final)


# Replace NA values with group mean using ave

for (col in names(PhenoSoil_data_final)) {
  if (is.numeric(PhenoSoil_data_final[[col]])) {  # Ensure it's a numeric column
    for (loc in unique(PhenoSoil_data_final$Field_Location)) {
      # Calculate the mean of the column for the specific Field_Location, excluding NAs
      group_mean <- mean(PhenoSoil_data_final[PhenoSoil_data_final$Field_Location == loc, col], na.rm = TRUE)

      # Replace NA values with the calculated mean within that group
      PhenoSoil_data_final[PhenoSoil_data_final$Field_Location == loc & is.na(PhenoSoil_data_final[[col]]), col] <- group_mean
    }
  }
}

write.table(PhenoSoil_data_final,file = 'PhenoSoil_data_final.csv',col.names = T,row.names = F,sep = '\t',quote = F)

####for test set#####

##for EC data
Test_data_EC <- Test_data_EC %>%
  distinct(Env, .keep_all = TRUE)


TestPhenoEC_data_final <- Test_data %>%
  left_join(., Test_data_EC, by="Env") %>%
  filter(rowMeans(is.na(.)) < 0.5)


summary(TestPhenoEC_data_final$CumHI30_pFloFla)


write.table(TestPhenoEC_data_final,file = 'TestPhenoEC_data_final.csv',col.names = T,row.names = F,sep = '\t',quote = F)

##taking the mean of the weather data
Weather_data_test <- aggregate(Test_data_weather_year[,3:18], by=list(Test_data_weather_year$Env), function(x) mean(x, na.rm = TRUE))
colnames(Weather_data_test)[1] <- "Env"

Weather_data_test <- Weather_data_test %>%
  mutate(Env = case_when(
   Env== "MOH1_1_2018" ~ "MOH1_2018",
   Env== "MOH1_1_2020" ~ "MOH1_2020",
    Env== "MOH1_2_2018" ~ "MOH1_2018",
   Env== "MOH1_2_2020" ~ "MOH1_2020",
   TRUE ~ Env
    ))

TestPhenoweather_data_final <- Test_data %>%
  left_join(., Weather_data_test, by="Env")

summary(TestPhenoweather_data_final)

write.table(TestPhenoweather_data_final,file = 'TestPhenoweather_data_final.csv',col.names = T,row.names = F,sep = '\t',quote = F)


##for metadatatest
Meta_datatest <- Test_data_meta %>%
  separate(Env, c("Field_Location", "Year"),"_", remove=FALSE)%>%
  filter(Field_Location %in% test_environment)%>%
  select(c("Env","Treatment","City","Previous_Crop","Irrigated","Latitude_of_Field_Corner_#1 (lower left)",
           "Longitude_of_Field_Corner_#1 (lower left)","Latitude_of_Field_Corner_#2 (lower right)",
           "Longitude_of_Field_Corner_#2 (lower right)","Latitude_of_Field_Corner_#3 (upper right)",
           "Longitude_of_Field_Corner_#3 (upper right)","Latitude_of_Field_Corner_#4 (upper left)",
           "Longitude_of_Field_Corner_#4 (upper left)","Weather_Station_Latitude (in decimal numbers NOT DMS)",
           "Weather_Station_Longitude (in decimal numbers NOT DMS)"))

Metaltest_others=Meta_datatest %>% select(c("Env","Treatment","City","Previous_Crop","Irrigated"))

Metaltest_final <- Meta_datatest %>%
  select(-c(Treatment, City, Previous_Crop, Irrigated))

TestPhenometal_data_final <- Test_data %>%
  left_join(., Metaltest_final, by="Env")

summary(TestPhenometal_data_final)

write.table(TestPhenometal_data_final,file = 'TestPhenometal_data_final.csv',col.names = T,row.names = F,sep = '\t',quote = F)

###for soil datatest
Test_data_soil<- Test_data_soil %>%
  separate(Env, c("Field_Location", "Year"),"_", remove=FALSE)

TestSoil_data <- Test_data_soil[,-c(2:6,30:36)] %>%
  distinct(Env, .keep_all = TRUE)

TestPhenoSoil_data_final <- Test_data %>%
  left_join(., TestSoil_data, by="Env")

summary(TestPhenoSoil_data_final)

# write.table(TestPhenoSoil_data_final,file = 'TestPhenoSoil_data_final.csv',col.names = T,row.names = F,sep = '\t',quote = F)

######for genomic analysis####

##Missing marker imputation, replacing missing marker with the mean values

SNP_data_numeric <- tibble(fread(file = "/usr/users/osatohanmwen/G2F_competition_2024/Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt"))
colnames(SNP_data_numeric)[1] <- "Genotype"

genID = SNP_data_numeric$Genotype

SNP_data_numeric_1 <- SNP_data_numeric[,-1] * 2


mean_val <- round(colMeans(SNP_data_numeric_1, na.rm=TRUE))

for (i in colnames(SNP_data_numeric_1)){
  SNP_data_numeric_1[,i][is.na(SNP_data_numeric_1[,i])] <- mean_val[i]
}

SNP_data_numeric_1 <- data.frame(Hybrid=SNP_data_numeric$Genotype, SNP_data_numeric_1)

table(is.na(SNP_data_numeric_1))

#write.table(SNP_data_numeric_1,file = 'Genotype_True.csv',col.names = T,row.names = F,sep = '\t',quote = F)

#### 2NPmatrices ###

### Van Raden(2008) deviation of  Genomic relationship matrix (additive)####
SNP_data_G <- as.data.frame(SNP_data_numeric_1[,-1])%>%
  mutate_if(is.character, as.numeric)

alelleFreq <- function(x, y) {
  (2 * length(which(x == y)) + length(which(x == 1)))/(2 *
                                                         length(which(!is.na(x))))
}


Frequency <- cbind(apply(SNP_data_G, 2, function(x) alelleFreq(x,0)),
                   apply(SNP_data_G, 2, function(x) alelleFreq(x, 2)),
                   colMeans(SNP_data_G)/2)


FreqP <- matrix(rep(Frequency[, 2], each = nrow(SNP_data_G)),
                ncol = ncol(SNP_data_G))

MyTwoPQ <- sum((2 *Frequency[, 1]) * Frequency[, 2])

SNP_data_G <- SNP_data_G- 2 *FreqP


SNP_data_G[is.na(SNP_data_G)] <- 0

SNP_data_G  <- as.matrix(SNP_data_G)

Genomic_addev_matrix <- data.frame(Hybrid=SNP_data_numeric_1[,1],SNP_data_G)


### Vitezica (2012) Genomic relationship matrix (dominance)####
SNP_data_D <- as.data.frame(SNP_data_numeric_1[,-1])%>%
  mutate_if(is.character, as.numeric)

dom_twoPQ <- 2*(FreqP[1,])*(1-FreqP[1,])

SNP_data_D[is.na(SNP_data_D)] <- 0

SNP_data_D <- (SNP_data_D==0)*-2*(FreqP^2) +
  (SNP_data_D==1)*2*(FreqP)*(1-FreqP) +
  (SNP_data_D==2)*-2*((1-FreqP)^2)

Genomic_dodev_matrix <- data.frame(Hybrid=SNP_data_numeric_1[,1],SNP_data_D)

write.table(Genomic_addev_matrix,file = "Genomic_addev_matrix.csv",col.names = T,row.names = F,sep = '\t',quote = F)
write.table(Genomic_dodev_matrix,file = "Genomic_dodev_matrix.csv",col.names = T,row.names = F,sep = '\t',quote = F)

