Machine Learning Model for Predicting Road Network Disruptions
During Heavy Precipitation: A Case Study on the Kerala Floods 2018
A dissertation submitted to the University of Manchester for the
degree of MSc Geographical Information Science in the Faculty of Humanities
2023
Student ID: 11161619
School of Environment, Education and Development
Word Count: 9031

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Table of Contents
List of Illustrations ......................................................................................................................................... 3
Abstract ......................................................................................................................................................... 5
Acknowledgements ....................................................................................................................................... 6
Declaration .................................................................................................................................................... 7
Intellectual Property Statement ................................................................................................................... 7
1. Introduction .......................................................................................................................................... 8
1.1 Introduction to the Global Flooding Crisis .......................................................................................... 8
1.2 The Intersection of Floods and Urban Infrastructure ......................................................................... 8
1.3 Towards Resilient Infrastructure ........................................................................................................ 9
1.4 Leveraging Modern Predictive Techniques ....................................................................................... 10
1.5 This Study's Focal Point - The 2018 Kerala Floods ............................................................................ 10
1.6 Objective and Significance of the Research ...................................................................................... 11
2 Literature Review ................................................................................................................................ 11
2.1 Importance of Flood Prediction ........................................................................................................ 11
2.2 Diverse Approaches to Flood Prediction........................................................................................... 12
2.3 Machine Learning in Flood Prediction .............................................................................................. 13
2.4 Predictive Features used in ML based Flood Models ....................................................................... 14
2.5 Data Imbalance and its Rectification ................................................................................................ 16
2.6 Advantages of ML Based Models ...................................................................................................... 16
2.7 Kerala Floods 2018: A Case Study ..................................................................................................... 17
3. Methodology ....................................................................................................................................... 19
3.1 Choice of Study Area: Ernakulam and Idukki Districts, Kerala .......................................................... 19
3.2 Data Acquisition ................................................................................................................................ 22
3.3 Data Integration & Feature Extraction.............................................................................................. 27
1

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
3.4 Model Selection and Rationale ......................................................................................................... 30
3.5 Model Validation and Testing ........................................................................................................... 31
4. Results ................................................................................................................................................. 32
4.1 Feature Extraction ............................................................................................................................. 32
4.2 Logistic Regression Statistics for Feature Selection .......................................................................... 34
4.3 Recursive Feature Elimination Results .............................................................................................. 36
4.4 Selected Features .............................................................................................................................. 37
4.5 Model Performance .......................................................................................................................... 38
5. Discussions .......................................................................................................................................... 42
5.1 Vulnerability Assessment of Road Network ..................................................................................... 42
5.2 Creating an Algorithm to Predict Flooding and Feature Importance ............................................... 43
5.3 Validating the Algorithm with Data from the 2018 Floods ............................................................... 43
6. Conclusions ......................................................................................................................................... 44
7. References .......................................................................................................................................... 47
2

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
List of Illustrations
List of Figures
Figure 1: The study area comprising the districts of Ernakulam and Idukki in the state of Kerala, India. The
basemap is the courtesy of ESRI 2023. ....................................................................................................... 20
Figure 2: Road networks in the study area divided into different classes. The basemap is the courtesy of
ESRI 2023..................................................................................................................................................... 23
Figure 3: Digital Elevation Model of the study area obtained from SRTM. The basemap is the courtesy of
ESRI 2023..................................................................................................................................................... 24
Figure 4: Precipitaion raster of the study area. The basemap is the courtesy of ESRI 2023. ...................... 25
Figure 5: Drainage networks in the study area. The basemap is the courtesy of ESRI 2023. ..................... 25
Figure 6: Raster showing impervious pixels in the study area. The basemap is the courtesy of ESRI 2023.
.................................................................................................................................................................... 26
Figure 7: Raster showing water pixels in the study area. The basemap is the courtesy of ESRI 2023. ....... 27
Figure 8: Slope raster obtained from the DEM of the study area. The basemap is the courtesy of ESRI
2023. ........................................................................................................................................................... 28
Figure 9: Snippet of Python script for RFE. ................................................................................................. 29
Figure 10: Snippet of Python script for getting statistics from the Logistic Regression Model. ................. 30
Figure 11: Flow chart showing the methodology. ....................................................................................... 32
Figure 12: Summary statistics from the Logistic Regression model used for feature selection. ................. 34
Figure 13: Results from Recursive Feature Elimination. ............................................................................. 36
Figure 14: Histograms showing distribution of mean_slope, mean_precipitation, imperviousness and
distance_to_drainage.................................................................................................................................. 38
Figure 15: Histograms showing distribution of min_elevation, min_slope and max_slope. ...................... 38
Figure 16: Chart showing the distribution of flooded (1) and non-flooded (0) roads in the dataset. ........ 39
Figure 17: Confusion Matrix of the model’s performance. ......................................................................... 40
Figure 18: ROC curve of the model’s performance. .................................................................................... 41
List of Tables
Table 1: The different classes of roads in the dataset. ................................................................................ 32
3

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Table 2: Summary statistics of min_elevation, max_elevation, min_slope, min_precipitation and
max_precipitation. ...................................................................................................................................... 33
Table 3: Summary statistics of mean_elevation, mean_slope, mean_precipitation, imperviousness, and
distance_to_drainage.................................................................................................................................. 34
Table 4: Significance of Predictors in the Flood Model. .............................................................................. 36
Table 5: The mean values of all the characteristics used for training the model. ....................................... 39
Table 6: Metrics derived from confusion matrix. ........................................................................................ 41
4

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Abstract
In the contemporary era, the frequency and severity of global flooding events have alarmingly
intensified. A predominant reason behind this escalating trend is human intervention in natural
landscapes, resulting in significant modifications to indigenous drainage systems. This, combined
with the intensifying ramifications of global climate change, has exacerbated the frequency and
intensity of these catastrophic events. One of the most affected facets of urban infrastructure due
to these flooding events are the road networks. These networks serve as critical lifelines for
transportation and are paramount for swift emergency responses.
The primary focus of this study is to emphasize the pressing need for road networks that are not
only robust but also resilient. It is imperative that these networks are designed and maintained
with a vision that anticipates potential flood threats, ensuring that they can swiftly adapt to and
recover from any disruptions caused by these inundations. To address this challenge, this paper
presents a machine learning-driven model that predicts the likelihood of road flooding during
episodes of heavy rainfall.
While traditional hydrological models have been cornerstones in flood prediction and analysis,
they sometimes grapple with accurately depicting the complex interplay of factors leading to
floods. This limitation underscores the potential of machine learning models, which are inherently
adept at handling nonlinearities and modelling intricate relationships within vast datasets.
Tapping into the power of these advanced algorithms, this research meticulously analyses
historical flood data to train and refine the model. The objective is to enhance the model's
precision, thereby facilitating accurate flood predictions. This approach is geared towards
minimizing the adverse impacts of floods on human life and infrastructural assets.
5

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Acknowledgements
I would like to express my profound gratitude and appreciation to my dissertation supervisor, Dr.
Matt Tomkinson. His deep expertise, tireless dedication, and insightful feedback have been
instrumental in shaping and guiding this research. Dr. Tomkinson's mentorship has not only
enriched my academic experience but has also provided me with valuable life lessons, inspiring
me to achieve more than I ever thought possible. The depth of his knowledge and his
commitment to nurturing my academic potential have been truly invaluable throughout the
process of completing this dissertation.
Equally, I must convey my heartfelt appreciation to my family and friends who have been my
pillars of strength throughout my time at university. Their unwavering faith in me, even during
times of doubt, and their relentless encouragement have been a source of boundless inspiration.
The love, understanding, and patience they have shown have kept me grounded and focused,
reinforcing my aspirations and endeavours. I am profoundly indebted to them for their role in
shaping my journey and for being my guiding light during challenging times. I cherish their
support, and their unwavering belief in my capabilities means the world to me.
6

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Declaration
No portion of the work referred to in the dissertation has been submitted in support of an
application for another degree or qualification of this or any other university or other institute of
learning.
Intellectual Property Statement
i) The author of this dissertation (including any appendices and/or schedules to this dissertation)
owns certain copyright or related rights in it (the “Copyright”) and s/he has given The University
of Manchester certain rights to use such Copyright, including for administrative purposes.
ii) Copies of this dissertation, either in full or in extracts and whether in hard or electronic copy,
may be made only in accordance with the Copyright, Designs and Patents Act 1988 (as amended)
and regulations issued under it or, where appropriate, in accordance with licensing agreements
which the University has entered into. This page must form part of any such copies made.
iii) The ownership of certain Copyright, patents, designs, trademarks and other intellectual
property (the “Intellectual Property”) and any reproductions of copyright works in the
dissertation, for example graphs and tables (“Reproductions”), which may be described in this
dissertation, may not be owned by the author and may be owned by third parties. Such
Intellectual Property and Reproductions cannot and must not be made available for use without
the prior written permission of the owner(s) of the relevant Intellectual Property and/or
Reproductions.
iv) Further information on the conditions under which disclosure, publication and
commercialisation of this dissertation, the Copyright and any Intellectual Property and/or
Reproductions described in it may take place is available in the University IP Policy (see The
University of Manchester Intellectual Property Policy, in any relevant Dissertation restriction
declarations deposited in the University Library, The University Library’s regulations (see
http://www.library.manchester.ac.uk/about/regulations/) and in The University’s Guidance for
the Presentation of Dissertations.
7

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
1. Introduction
1.1 Introduction to the Global Flooding Crisis
Increasing flooding events around the globe have become a growing concern in recent years.
Human activities and their impact on natural and hydrological processes are recognized as
significant contributors to this trend (Ceola et al., 2014). Urbanization and changes in land use
also play a role in the increasing flooding events (Rentschler et al., 2023). The expansion of
cities and the conversion of natural landscapes into impervious surfaces, such as concrete and
asphalt, reduce the ability of the land to absorb rainfall, leading to increased surface runoff
and a higher likelihood of flooding (Tellman et al., 2021; Andreadis et al., 2022). The
encroachment of human settlements into flood-prone areas further exacerbates the risk, as
more people and infrastructure are exposed to flood hazards (Baldassarre et al., 2010).
Climate change is also playing a role, with rising global temperatures leading to more intense
rainfall events and an increased likelihood of flooding (Frederikse et al., 2022). Rising global
temperatures also increase water vapour in the atmosphere, leading to more intense rainfall
and an elevated risk of flooding (Trenberth, 2011).
Furthermore, changes in hydrological systems, such as alterations in river channels and the
construction of dams and levees, can impact the natural flow of water and increase the risk
of flooding (Lendering et al., 2015). Poor land management practices, such as deforestation
and improper soil conservation, can also contribute to increased runoff and soil erosion,
further exacerbating the risk of flooding (Madsen et al., 2014).
The impacts of increasing flooding events are significant and wide-ranging. They include
damage to infrastructure, loss of lives, displacement of populations, and economic losses
(Kundzewicz et al., 2013).
1.2 The Intersection of Floods and Urban Infrastructure
Effects of floods on urban infrastructure can be significant and wide-ranging. Floods can cause
damage to buildings, roads, bridges, and other critical infrastructure, leading to disruptions in
communication, and essential services (Huong & Pathirana, 2013). The force of floodwaters
can erode and weaken structures, compromising their stability and integrity (Hettiarachchi et
8

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
al., 2018). Floods can also damage electrical systems, water supply networks, and sewage
systems, further exacerbating the impact on urban infrastructure (Song et al., 2022).
Furthermore, floods can disrupt the operation of critical infrastructure systems, such as power
plants, hospitals, and transportation networks. Power outages can occur due to flooding of
electrical substations and damage to power distribution infrastructure (Ramos & Besharat,
2021). Hospitals and healthcare facilities may be forced to evacuate or operate at reduced
capacity, impacting the provision of essential medical services (Guan et al., 2021).
Flooding events have significant impacts on roads and transportation networks. During
flooding, roads can become submerged or washed away, rendering them impassable and
disrupting transportation systems (Pant et al., 2017). The damage to roads can range from
surface erosion and potholes to complete structural failure, depending on the intensity and
duration of the flood event. Floodwaters can also undermine the stability of road foundations,
leading to sinkholes and further compromising the integrity of the transportation network
(Pant et al., 2017). Flooded roads and railways can disrupt transportation systems, hindering
emergency response efforts and impeding the movement of people and goods (Nasiri et al.,
2016).
1.3 Towards Resilient Infrastructure
Hence, the resilience of road networks to flooding is crucial for ensuring the functionality and
continuity of transportation systems in urban areas. Resilience refers to the ability of a system
to anticipate, absorb, adapt, recover, and learn from disturbances or shocks (Kong et al. 2019).
In the context of road networks, resilience entails the capacity to withstand and recover from
flood events, minimizing disruptions to transportation services. Resilient road networks are
designed and managed to mitigate the impacts of flooding. This includes implementing
measures such as proper drainage systems, elevated road sections, and flood-resistant
materials in construction (Li et al., 2019). Additionally, advanced modelling and forecasting
techniques can help predict flood impacts on road networks, enabling proactive planning and
response strategies (Wang et al., 2019) which can help the road networks to adapt to events
of shock.
9

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
1.4 Leveraging Modern Predictive Techniques
Traditional hydrological models have several limitations in capturing the complex dynamics of
flood processes. One limitation is their simplified representation of the hydrological system,
which often assumes uniform and homogeneous conditions across the entire watershed (Troy
et.al, 2015). This oversimplification may lead to inaccurate predictions, especially in
heterogeneous landscapes with varying soil types, land cover, and topography (Chen et al.,
2021).
Another limitation is the reliance on historical data and stationary assumptions. Traditional
models assume that the statistical properties of hydrological variables, such as rainfall and
streamflow, remain constant over time (Read & Vogel, 2015). However, climate change and
land use changes can introduce nonstationarity, causing shifts in the frequency and intensity
of extreme events (Read & Vogel, 2015). Traditional models may fail to capture these changes,
leading to biased predictions and inadequate flood risk assessments.
To overcome these limitations, there is a growing interest in using machine learning (ML)
techniques to model flood processes. ML models have been increasingly used in predicting
floods due to their advantages in accuracy, efficiency, and convenience (Lu et al., 2022). These
models have shown promise in modelling nonlinear relationships and improving forecasting
results (Yang & Chang, 2020; Mosavi et al., 2018). Models such as artificial neural networks
(ANNs) and support vector machines, have been applied to runoff forecasting and flood
prediction (Lu et al., 2022; Mosavi et al., 2018). By analysing historical data and identifying
patterns, machine learning algorithms can provide timely forecasts of rainfall and flood hazard
variables, aiding in flood preparedness and risk reduction (Chen, 2021).
This study therefore aims at creating a machine learning model that can predict disruptions
in road networks during heavy precipitation.
1.5 This Study's Focal Point - The 2018 Kerala Floods
The urgency and significance of this research become even more palpable when considering
real-world scenarios, such as the 2018 floods in Kerala, India. With extremely high rainfall of
2346.6 mm from June to August, the rains resulted in severe flooding in 13 districts of Kerala
10

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
(Babu & Leno, 2020). The heavy impact of the floods on every sector, including animal
husbandry and agriculture, highlights the widespread devastation caused by the event (V.L.,
2021; Veerakumaran & Santhi, 2019). The severity of the 2018 floods in Kerala has resulted in
extensive damage to roads, making them impassable and affecting the overall connectivity of
the region (Mishra et al., 2018; Senan et al., 2022).
1.6 Objective and Significance of the Research
Given these challenges, this project aims to create a model capable of predicting road
disruptions during intense precipitation events. The project plans to achieve this through the
following objectives:
1. To assess the vulnerability of road networks to flooding, focusing on factors such as
elevation, slope, permeability and distance to drainage.
2. To fine-tune a machine learning-based algorithm that can accurately forecast the
likelihood of road flooding during heavy precipitation events.
3. To validate the algorithm using historical flood data, particularly from the 2018 floods in
Kerala, ensuring its robustness and reliability in real-world scenarios.
Accurate predictions empower authorities and communities to plan and deploy resources
efficiently. Also, as someone who has experienced the floods in this study personally, it is not
just about maintaining transportation—it is about fortifying the resilience of an entire
community against the destructive power of floods.
2 Literature Review
2.1 Importance of Flood Prediction
The importance of flood prediction cannot be overstated, as it plays a crucial role in mitigating
the adverse impacts of floods and improving disaster management strategies. Floods are
among the most destructive natural disasters, causing extensive damage and loss of life
(Mosavi et al., 2018). The costs associated with flood damage and recovery can be substantial,
with potential annual losses of up to 9.3% of global gross domestic product (Hinkel et al.,
2014). Therefore, accurate flood prediction models are crucial for risk reduction, policy
development, and minimizing the loss of human life and property damage associated with
11

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
floods (Mosavi et al., 2018). Flood prediction also plays a vital role in urban planning and
infrastructure development. By accurately predicting flood events, city planners can identify
areas at high risk of flooding and implement appropriate measures to mitigate the impact.
This may include constructing flood barriers, improving drainage systems, or implementing
land-use regulations to prevent construction in flood-prone areas (Danso et al., 2020).
Accurate flood prediction can also inform the design and construction of critical
infrastructure, such as bridges and dams, ensuring they are built to withstand potential flood
events (Jafarzadegan et al., 2023). Furthermore, flood prediction models can help in the
management of water resources. Water resource managers can make informed decisions
about water allocation and reservoir operations by accurately predicting flood events. This
can help prevent overflows and ensure the efficient use of water resources during flood
events (Zhang et al., 2018).
2.2 Diverse Approaches to Flood Prediction
Several different flood prediction models are available that utilize various techniques and
approaches. These models aim to accurately predict the occurrence and severity of floods to
support decision-making and mitigate the impacts of flooding. Some of the notable flood
prediction models include:
1. Machine Learning Models: Machine learning techniques, such as artificial neural networks
(ANN), support vector regression (SVR), and random forest, have been widely used in flood
prediction (Mosavi et al., 2018). These models utilize historical flood data and other relevant
variables to train the model and make predictions. They have shown promising results in
accuracy and performance (Esfandiari et al., 2020).
2. Hydrological Models: Hydrological models, such as the Hydrologic Engineering Center's
Hydrologic Modelling System (HEC-HMS), use mathematical equations to simulate the
hydrological processes that lead to flooding (Banihabib et al., 2020). These models predict
flood events using rainfall, soil moisture, and river flow. They are commonly used for riverine
flood prediction.
12

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
3. Statistical Models: Statistical models, such as the index flood method and quantile
regression, estimate flood quantiles and predict flood events (Ishak et al., 2011). These
models analyse historical flood data and identify patterns and relationships between flood
variables. They are often used when at-site recorded flood data are limited.
4. Data-Driven Models: Data-driven models, such as Gaussian processes and deep learning
models, utilize large datasets to make predictions (Brockkötter et al., 2021). These models can
capture complex relationships and patterns in the data and provide accurate flood predictions
(Qian, 2019). They have shown promising results in real-time flood forecasting and early
warning systems (Wu et al., 2020).
5. Ensemble Models: Ensemble models combine multiple individual models to improve the
accuracy and reliability of flood predictions (Choubin et al., 2019). These models take
advantage of the strengths of different models and reduce the uncertainties associated with
individual models (Berkhahn et al., 2019). Ensemble models can include a combination of
machine learning, hydrological, and statistical models.
6. Remote Sensing Models: Remote sensing data, such as satellite imagery and radar data,
can be used in flood prediction models (Schumann et al., 2009). These models utilize the
information from remote sensing technologies to estimate flood extent and monitor flood
dynamics (Mah et al., 2012). Remote sensing models can provide valuable input data for other
flood prediction models.
7. Physics-Based Models: Physics-based models, such as the Saint Venant equations, simulate
the physical processes of water flow and flooding (Elhanafy & Copeland, 2007). These models
consider the fundamental principles of fluid dynamics and hydraulic behaviour to predict
flood events (Lhomme et al., 2008). Physics-based models are often used for detailed flood
simulations and floodplain inundation mapping.
2.3 Machine Learning in Flood Prediction
In the realm of flood prediction, various machine learning models have gained prominence.
The Support Vector Machine (SVM) stands out for its versatility in handling both classification
and regression, finding application in flood susceptibility mapping and forecasting (Li et
13

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
al.,2019; Liu et al., 2021; Tehrany et al., 2015; Tehrany et al., 2014; Ha et al., 2021). The
Random Forest (RF), amalgamates numerous decision trees, enhancing the accuracy of flood
predictions (Khalaf et al., 2020; Sampurno et al., 2022; Granata et al., 2016; Li et al., 2016).
Artificial Neural Networks (ANN), drawing inspiration from human neural structures, are
recognized for their prowess in deciphering intricate nonlinear relationships (Atashi et al.,
2022; Tayfur et al., 2018; Hu et al., 2021). Decision trees, with their decision-making tree-like
frameworks, have been instrumental in flood forecasting (Choubin et al., 2019; Antonetti et
al., 2019). Ensemble models, which merge algorithms like SVM, RF, and ANN, prioritize
prediction precision (Jia et al., 2022; Rahman et al., 2021). The nonlinear regression approach,
Multivariate Adaptive Regression Splines (MARS), is lauded for addressing complex
relationships in flood dynamics (Costache & Bui, 2020). Deep learning models, encompassing
Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), have been
impactful given their capacity to process voluminous data and identify spatial-temporal flood
patterns (Chen et al., 2023; Kondo et al., 2023). Lastly, hybrid models, which merge varied
algorithms or couple machine learning with other models like hydrological ones, are tailored
to harness the merits of diverse methodologies, aiming for enhanced flood prediction
accuracy (Liu et al., 2021; Pham et al., 2017; Xie & Shen, 2021).
2.4 Predictive Features used in ML based Flood Models
In flood prediction model research, a plethora of features have been utilized to optimize their
predictive capacities. Meteorological data, encompassing aspects like rainfall patterns and
humidity, provides a foundational understanding of precipitation dynamics leading to floods
(Ward et al., 2014; Chang et al., 2020). Using high-resolution precipitation data is crucial in
flood risk analysis. Precipitation plays a significant role in flooding, and accurate information
about its spatial and temporal distribution is essential for understanding flood patterns and
assessing flood risk (Shiu et al., 2012).
Complementing this, hydrological variables, such as river flow rates and discharge
measurements, elucidate the riverine responses to precipitation events (Tamiru & Wagari,
2021; Tabarestani & Afzalimehr, 2021). The terrain's influence on flood dynamics is assessed
through topographic data, highlighting elevation and slope gradient factors that direct water
14

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
flow (Li et al., 2021; Wang et al., 2021). By analysing the slope of roads derived from Digital
Elevation Models (DEM), we can identify areas prone to water accumulation and potential
drainage issues during flood events (Ferencevic & Ashmore, 2011). The elevation information
extracted from DEMs allows for the identification of low-lying areas and areas at risk of
inundation during floods (Hashemi-Beni et al., 2018). This information is essential for
understanding the spatial extent and severity of flood events and for developing effective
flood risk management strategies (Hashemi-Beni et al., 2018).
Land use and land cover data, which details the spatial arrangement of urban areas, forests,
and other terrains, further influences flood genesis by affecting infiltration and runoff
processes (Li et al., 2021). Soil properties, including moisture content and infiltration capacity,
contribute to runoff generation dynamics (Wang et al., 2021; Tabarestani & Afzalimehr, 2021).
One study conducted in Shanghai, China, evaluated the impact and risk of pluvial flash floods
on the intra-urban road network (Yin et al., 2016). The results showed that roads closer to
drainage networks experienced a greater impact from flooding. This finding is consistent with
studies conducted in other cities, such as Portland and Boston, where urban transportation
networks also demonstrated a higher vulnerability to flooding (Yin et al., 2016).
A study focused on the variation of surface runoff and soil moisture content in the subgrade
of permeable pavement structures (Hou et al., 2020) found that permeable roads can greatly
alleviate the pressure of urban drainage and reduce the risk of storms and floods.
Advancements in technology, notably remote sensing, offer insights on vegetation and surface
water distribution, enhancing the spatial comprehension of flood variables (Munawar et al.,
2022). Historical flood data, encompassing prior flood locations and characteristics, serves as
an essential reference to discern patterns and relationships (Mistry & Parekh, 2022;
Tabbussum & Dar, 2021). Lastly, other contingent factors such as anthropogenic influences
and dam operations might be factored into models depending on the specific research context
(Ward et al., 2014; Tabbussum & Dar, 2021).
15

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
2.5 Data Imbalance and its Rectification
The reality of most natural disaster datasets is their imbalance. The dominance of "non-event"
data (in this case, non-flooded roads) can skew model training, leading to biased predictions
(Rahman & Davis, 2013). This is because the models are more likely to be biased towards the
majority class, as they have more examples to learn from (Chawla et al., 2002). As a result,
the minority class may be overlooked or misclassified, leading to poor performance in real-
world applications (??M?EK & DA?, 2022).
To address the challenges posed by class imbalance, researchers have proposed various
techniques. One approach is to use sampling methods, such as oversampling the minority
class or undersampling the majority class, to balance the class distribution (Chawla et al.,
2002). Oversampling techniques, such as Synthetic Minority Over-sampling Technique
(SMOTE), generate synthetic examples of the minority class to increase its representation in
the dataset (Chawla et al., 2002). Undersampling techniques, on the other hand, reduce the
number of examples from the majority class to match the number of examples in the minority
class (Bach et al., 2017). These techniques aim to create a more balanced training set, allowing
the model to learn from both classes more effectively.
2.6 Advantages of ML Based Models
Machine learning models present a range of superiorities in flood prediction compared to
conventional techniques. Firstly, they excel in discerning complex relationships and
pinpointing intricate patterns and nonlinear interactions, which standard models might often
miss (Atashi et al., 2022). Anchored in a data-driven paradigm, they harness extensive
historical flood records, enabling them to refine and enhance predictions iteratively. This
adaptability extends to diverse flood types and geographic areas, and by tailoring them to
specific datasets, they can be aptly aligned with a region's unique attributes (Kohansarbaz et
al., 2022). An integrative capability allows these models to amalgamate data from diverse
avenues like remote sensing, meteorological records, and past flood events
(Sankaranarayanan et al., 2019). Addressing one of the recurrent challenges, these models
capably manage missing or fragmentary data, ensuring consistent forecasting despite data
gaps (Tayfur et al., 2018). Emphasizing their adaptability, they are scalable, proficiently
16

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
handling extensive tasks, a feature pivotal for prominent urban localities or expansive river
catchments (Kondo et al., 2023).
2.7 Kerala Floods 2018: A Case Study
Kerala, located in India, has consistently faced floods over the years. A particularly severe
flood event hit the region in August 2018, affecting around 23 million individuals. This disaster
was regarded as Kerala's most detrimental since the Great flood of 1924 (Divakaran et al.,
2019). The flood's intensity resulted from a combination of substantial rainfall and the release
of water from several dams, leading to vast destruction across infrastructure, agriculture, and
many people's livelihoods (Kumar et al., 2020). Notably, the socio-economic well-being of
dairy farmers was significantly disrupted by this flood event (V.L., 2021).
Research suggests that the extreme precipitation and flooding in Kerala in 2018 were
influenced by climate change (Dhasmana et al., 2023). Climate models indicate a significant
increase in rainfall and moisture flux over the peninsula's southern part, contributing to the
flooding (Hunt & Menon, 2020). Another factor contributing to the floods was changing land
use and land cover (LULC) (Dixit et al., 2022). Improper dam operations also affected the
flooding (Dixit et al., 2022). The opening of dams to release excess water exacerbated the
flooding in the region (Neeraj et al., 2020).
The effects of the flood on the road infrastructure were severe. The floodwaters washed away
roads and caused landslides, making many roads impassable (Thakur et al., 2020). This
significantly impacted the transportation network in the affected areas, hindering the
movement of people and goods. The flood also had long-term effects on the road
infrastructure. The damage caused by the flood required extensive repairs and reconstruction
of roads and bridges. The cost of repairing the road infrastructure in Kerala after the flood
was estimated to be substantial (Mishra et al., 2018).
In the subsequent years, Kerala confronted back-to-back flood challenges, especially in 2018
and 2019 (Joseph, 2022). Such events accentuated the requirement for strengthened disaster
management strategies within the state. There have been discrepancies between predicted
flood-vulnerable zones and the actual regions that got affected, suggesting a pressing need to
17

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
integrate effective disaster mitigation plans in Kerala's Municipal Corporations' planning
documentation (Shankar & Bindu, 2021).
Multiple research initiatives have been undertaken to gain insights into the root causes of
Kerala's floods and to propose potential countermeasures. Kumar et al. (2020) dissected the
role of meteorological elements, such as low-pressure systems and dry air intrusions, in
exacerbating the August 2018 flood. Their findings suggested that both heavy rainfall and dam
water releases intensified the flooding. Furthermore, a different study concentrated on the
aftermath of climate change on Kerala's public health infrastructure during these floods,
underlining Kerala's susceptibility to climate-affected calamities (Varughese &
Purushothaman, 2021). Asim et al. (2019) shed light on the mental health of the affected,
noting a significant prevalence of post-traumatic stress disorder (PTSD) among the flood
victims. This is further echoed by Jose & Fenn (2021) who discussed the profound
psychological implications on farmers, underscoring the role of resilience in their recovery
journey. Divakaran et al. (2019) ventured into the biological aftermath of the flood,
discovering intricate bacterial compositions and antibiotic resistance patterns in the
inundated areas. In pursuit of a comprehensive flood hazard analysis, Thakur et al. (2020)
synergized remote sensing, GIS, and hydrological models. Dhasmana et al. (2023) examined
the flood through the lens of climate change, providing evidence of its influence on the
disaster's severity and highlighting the urgency for adaptation strategies. Urban preparedness
surfaced as a theme in Shankar & Bindu (2021), emphasizing the imperative to embed disaster
mitigation into municipal planning. With public health repercussions at the forefront,
Varughese & Purushothaman (2021) spotlighted the vulnerability of Kerala's health systems
in the face of such climate-induced adversities. Exploring the economic realm, V.L., (2021)
focused on the socio-economic repercussions on dairy farmers, while Greeshma & Greeshma
(2023) assessed the economic tremors felt by the plantation sector. George et al. (2022)
tapped into the potential of the Hydrologic Engineering Centers Hydrologic Modelling System
(HEC-HMS), illustrating its efficacy in flood modelling. Meanwhile, Veerakumaran & Santhi
(2019) presented a narrative of the flood's impact on agriculture, and Pradeep & Rajesh
(2020) narrated the socio-economic disruptions endured by communities. Post-flood, Babu &
18

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Leno (2020) delved into the environmental facet, assessing the organic carbon status in flood-
impacted terrains.
Post the 2018 floods, research on Kerala's rehabilitation process indicated both successes and
challenges in implementing the "Build Back Better" (BBB) principle in recovery initiatives
(Neeraj et al., 2020). This underscores the significance of adopting an all-encompassing and
enduring post-flood recovery framework.
During these floods, social media emerged as a crucial tool in organizing and facilitating
disaster relief efforts, aiding in resource allocation, unifying relief activities, and voicing the
urgency for rescue missions (Joseph, 2022).
Lastly, to robustly counter flood-related challenges, Kerala needs to embrace a holistic flood
mitigation plan comprising well-strategized land utilization, water regulation, and
infrastructural advancements. Given that elements like changes in land usage, enhanced
impermeable areas, and evolving hydrological scenarios contribute to Kerala's flood
vulnerabilities, it's imperative to incorporate them in flood risk evaluation and counter-
strategies (Mishra et al., 2018; Sarkar & Mondal, 2019; Dang & Kumar, 2017).
3. Methodology
3.1 Choice of Study Area: Ernakulam and Idukki Districts, Kerala
Kerala, a state located in the southwestern region of India, is characterized by various
geographical features. The state is relatively small in size and is situated on the southern
peninsular region of India (Ramanarayanan & Rajeev, 2020). It is bordered by the Arabian Sea
on the west and is known for its beautiful coastline (Hunt & Menon, 2020). The state is also
home to the Western Ghats, a mountain range that runs parallel to the Arabian Sea and is
known for its rich biodiversity and scenic landscapes (Hunt & Menon, 2020). The Western
Ghats play a crucial role in the monsoon rainfall patterns of Kerala (Simon & Mohankumar,
2004).
Kerala experiences a tropical climate, with high temperatures and humidity throughout the
year (Lu et al., 2020). The state receives heavy rainfall during the monsoon season, which lasts
from June to September (Ananthakrishnan & Soman, 1988). The monsoon rains are vital for
19

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
the agriculture and ecosystem of the region. Kerala's rivers, such as the Periyar, Pamba, and
Bharathapuzha, are fed by these monsoon rains and play a significant role in the state's
irrigation and water supply (Simon & Mohankumar, 2004).
The coastal plains of Kerala are characterized by backwaters, which are a network of
interconnected canals, lakes, and lagoons. These backwaters are not only a unique
geographical feature but also a major tourist attraction, known for their scenic beauty and
houseboat cruising. The state is also known for its sandy beaches, such as Kovalam and
Varkala, which attract tourists from around the world (Hunt & Menon, 2020).
In addition to its coastal and mountainous regions, Kerala is also known for its rich
biodiversity. The state is home to numerous wildlife sanctuaries and national parks, including
Periyar Tiger Reserve, Wayanad Wildlife Sanctuary, and Silent Valley National Park (Hunt &
Menon, 2020). These protected areas are crucial for the conservation of various species of
flora and fauna. The study area map is shown in Figure 1.
Figure 1: The study area comprising the districts of Ernakulam and Idukki in
the state of Kerala, India. The basemap is the courtesy of ESRI 2023.
20

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
The state witnessed unprecedented floods in the year 2018 which primarily affected the
central parts of the state, mainly the districts of Ernakulam and Idukki, and resulted in the
inundation of approximately 1100 km2 of land (Vishnu et al., 2019). The flood was caused by
a combination of extreme rainfall and the release of water from reservoirs in the region
(Mishra et al., 2018). In Ernakulam district, which is located on the coastal plains of Kerala,
the heavy rainfall and rising water levels resulted in the submergence of low-lying areas,
including roads and bridges, leading to disruptions in transportation (Vishnu et al., 2019).
Similarly, in Idukki district, which is situated in the mountainous region of the Western Ghats,
the floods caused significant damage to the road infrastructure (Vishnu et al., 2019). The
heavy rainfall and landslides in the hilly terrain resulted in the blockage and collapse of roads,
making them impassable (Vishnu et al., 2019). Intense rainfall events, such as the 177.5 mm
of rainfall on August 17, 2018, in Idukki, led to overflowing reservoirs and dams in the highland
and midland areas (Vishnu et al., 2019).
The significance of using the 2018 Kerala floods as a study area for creating a model that can
predict road flooding during heavy precipitation lies in the severity and complexity of the
event. By studying this event, researchers can gain valuable insights into the factors that
contribute to the severity of flooding and develop a predictive model to anticipate and
mitigate the impacts of future flooding events on road infrastructure. Kerala's vulnerability to
extreme weather events, its high rainfall, and its complex topography make it an ideal study
area for understanding the dynamics of road flooding and developing effective strategies for
disaster management and urban planning (Singh et al., 2018). The insights from studying the
2018 Kerala floods can also be applied to other regions facing similar challenges, where heavy
precipitation and inadequate infrastructure can lead to road flooding and associated
disruptions (Ginkel et al., 2021). By developing a model that can accurately predict road
flooding during heavy precipitation, authorities can improve their preparedness and response
strategies, ultimately reducing the loss of life, property, and economic costs associated with
road infrastructure damage during floods (Lamb et al., 2019).
21

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
3.2 Data Acquisition
Data acquisition for flood risk analysis in developing nations like India poses several
challenges. One of the main challenges is the need for more accurate and reliable data
(Mansour, 2020). This is crucial for effective flood risk management and the development of
appropriate mitigation strategies. The availability of accurate data is essential for identifying
flood-prone areas, assessing vulnerability, and understanding the potential impacts of
flooding (Mansour, 2020). This was the major challenge faced in the data collection stage of
this research, as data about the 2018 floods were scarcely available. Based on literature, it
was decided that the training of the disruption likelihood model requires data about the
flooded roads and their characteristics like elevation, slope, permeability, precipitation and
distance to drainage, to predict the availability of the road for use. Since an exhaustive list of
this kind of data was unavailable, this research generated the data from remote sensing
products.
Utilizing geospatial data files, we can access intricate details about road networks, which
proves instrumental in assessing potential risks and impacts of floods on urban road systems
(Yin et al., 2016). Such files, when incorporated into flood modelling, enable a systematic
evaluation of the effects of flash flood events on these networks (Yin et al., 2016). Since there
was an unavailability of information about the affected road networks in the study area, this
research aimed at using shapefiles of the road networks so that they could be further used to
extract characteristics of each of the roads in the study area. The road network shapefiles
required for this work was obtained from Geofabrik. Geofabrik is a server that provides data
extracts from the OpenStreetMap (OSM) project (Haklay & Weber, 2008). OSM is a
collaborative mapping project that allows users to contribute and edit street maps (Haklay &
Weber, 2008). Geofabrik's server offers free downloads of OSM data extracts, which are
regularly updated (Abshirini & Koch, 2016). The data downloaded included shapefiles of
places, land use, transport, buildings etc. The road networks shapefile was extracted from this
data and was then clipped to the study area extent as shown in Figure 2 below.
22

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Figure 2: Road networks in the study area divided into different classes. The basemap is the
courtesy of ESRI 2023.
To acquire information about the elevation and slope of the road networks, a DEM provide by
the Shuttle Radar Topography Mission (SRTM), acquired from the OpenTopography portal was
used (Figure 3). The SRTM was a space mission conducted by NASA and the National Imagery
and Mapping Agency (NIMA) with the goal of creating a high-resolution DEM of the Earth's
surface (Farr et al., 2007). The mission was flown in February 2000 and collected data using a
radar instrument onboard the Space Shuttle Endeavour (Zandbergen, 2008). The SRTM
mission aimed to provide accurate and consistent elevation data on a global scale, covering
approximately 80% of the Earth's surface between 60°N and 56°S (Zandbergen, 2008).
23

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Figure 3: Digital Elevation Model of the study area obtained from SRTM.
The basemap is the courtesy of ESRI 2023.
The highest resolution precipitation data available for the study area was from Climate
Hazards InfraRed Precipitation with Station data (CHIRPS). CHIRPS is a satellite-based
precipitation dataset that provides high-resolution and long-term estimates of rainfall (Funk
et al., 2015). It combines infrared Cold Cloud Duration (CCD) observations with smart
interpolation techniques to create a global record of precipitation (Funk et al., 2015). The
CHIRPS dataset has relatively high spatial and temporal resolutions, making it suitable for
various applications, including drought monitoring, hydrological modelling, and climate
change studies (Dinku et al., 2018; Bayissa et al., 2017). The resolution of the data was about
6 km. Even though this is higher than the resolution of the DEM used in this study, this raster
was able to give some reliable information about the variability of precipitation in the study
area.
24

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Figure 4: Precipitaion raster of the study area. The basemap is the
courtesy of ESRI 2023.
To acquire data about the distance to the closest drainage network for each road, Geofabrik
server was used. The shapefiles acquired of the drainage network is shown in Figure 5.
Figure 5: Drainage networks in the study area. The basemap is the
courtesy of ESRI 2023.
25

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
A raster layer created by Tsinghua University was acquired from GEE which was then used to
identify how permeable each road network was. This dataset contains annual change
information of global impervious surface area from 1985 to 2018 at a 30m resolution (Gong
et al., 2020)
Figure 6: Raster showing impervious pixels in the study area. The
basemap is the courtesy of ESRI 2023.
Finally, data about which roads were flooded during the 2018 event was required so that it
could be used to train the model. The first attempt made was to acquire Landsat images from
Google Earth Engine (GEE) to identify flooded regions and then intersect them with the roads
to see which of them were flooded. But the challenge here was that almost all the images
from this period had very high cloud cover which made it impossible to identify the flooded
regions. This was a limitation of all the images acquired from an optical sensor. This is when
an approach was made to use Synthetic Aperture Radar (SAR) images. SAR images have the
ability to penetrate clouds and heavy rainfall, making them particularly valuable for flood
mapping and evaluation (Huang & Jin, 2020). So, a SAR image was acquired for the date of
21st August 2018 from ESA Copernicus which was then pre-processed in SNAP software. Then
26

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
a global thresholding approach was used to identify the water pixels (Figure 7) from the image
which was then used to identify the flooded roads.
Figure 7: Raster showing water pixels in the study area. The basemap is the
courtesy of ESRI 2023.
3.3 Data Integration & Feature Extraction
The zonal statistics as table tool in ArcGIS Pro was used to extract the elevation information
of the roads. The input feature class or zones was selected as the road networks shapefile and
the raster was selected as the DEM. Each road has a unique osm_id which can be used to
identify the road. This was used as the zone field which would be used to calculate the zonal
statistics. This tool was used to extract the mean, minimum and maximum elevation of each
road network.
The slope tool in ArcGIS Pro was then used to create raster layer from the dem with each pixel
showing the slope. The zonal statistics as table tool was then uses to extract the minimum,
maximum and mean slope of each of the roads.
27

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Figure 8: Slope raster obtained from the DEM of the study area. The basemap
is the courtesy of ESRI 2023.
The imperviousness raster obtained from GEE contained information on when each pixel in
the raster became impervious. This raster was reclassified so that all the impervious pixels
had a value of 1 and the rest had value of 0. This was done so that when the zonal statistics
as table tool would be used to get the sum of these pixels that come under each road, a higher
value would mean that a large region around the road is impervious.
Even though the study area is only two districts from the state of Kerala, there can be
variations in the amount of precipitation. To obtain information about the precipitation that
each road received, the zonal statistics as table tool was again used with the roads as the
zones and the precipitation layer as the input raster.
The drainage network data obtained from OSM was then used with the near tool in ArcGIS
Pro to identify the distance from each road to its nearest drainage source.
The zonal statistics as table tool was used on the water pixels layer created in SNAP to identify
flooded roads. The tool was used to calculate the sum of pixel values under each road. If the
28

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
sum came out to be 0, it would mean that there are no water pixels under the road and hence
the road was not flooded.
For each road in the network, characteristics like mean elevation, minimum elevation,
maximum elevation, mean slope, minimum slope, maximum slope, mean precipitation,
maximum precipitation, minimum precipitation, imperviousness and distance to drainage
were compiled a csv file. It was found that there was a very big imbalance in the data. To
rectify this, the oversampling method of SMOTE was employed. The advantage of using
SMOTE is that it does not result in a loss of useful data (Shi et al., 2022). Unlike undersampling
methods that discard instances from the majority class, SMOTE creates synthetic examples by
interpolating between existing minority class instances (Shi et al., 2022). This ensures that all
available information is utilized in the training process.
After balanced data was ensured, to find out the features that were significant in predicting
flood, Recursive Feature Extraction (RFE) and statistics from a logistic regression model were
used. Ding & Wilkins (2006) highlight that RFE is a well-studied method for reducing the
number of attributes used in prediction models or further analysis. By iteratively eliminating
features based on their importance, this technique helps to identify the most relevant
features and improve the efficiency and interpretability of the models. RFE was implemented
through the python script shown in Figure 9.
Figure 9: Snippet of Python script for RFE.
29

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
The significance of logistic regression coefficients can be assessed through hypothesis testing.
The p-values associated with the coefficients indicate whether the relationship between each
predictor and the outcome is statistically significant. A low p-value (typically below a
predetermined significance level, e.g., 0.05) suggests that the coefficient is significantly
different from zero, indicating a meaningful relationship (Kim et al., 2014). This was
implemented in this study using statsmodels in python as shown in Figure 10.
Figure 10: Snippet of Python script for getting
statistics from the Logistic Regression Model.
The features that were not significant were removed from the data frame based on the results
from these two implementations and the rest was used to train the model. This data was used
in python using statsmodels to perform logistic regression. This package provides detailed
statistical summaries, which are often useful for understanding the significance and fit of your
model
3.4 Model Selection and Rationale
This study has used a Logistic Regression Model. Logistic regression is a widely used statistical
modelling technique that is suitable for binary classification problems, such as predicting
flood occurrence or non-occurrence. It provides interpretable results by estimating the
probability of an event based on input variables Wei et al. (2009). Logistic regression has been
applied in flood susceptibility mapping, flood stage forecasting, and flood risk assessment
(Tehrany et al., 2014; Giovannettone et al., 2018). It is computationally efficient and can
handle large datasets, making it a practical choice for flood prediction models (Rahmati et al.,
2020). Logistic regression can also provide insights into the relative importance of input
variables in influencing flood occurrence (Giovannettone et al., 2018).
30

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Logistic regression provides interpretable results and is computationally efficient, but it
assumes a linear relationship between input variables and flood occurrence (Yaseen et al.,
2022).
3.5 Model Validation and Testing
Once models are trained, it's imperative to validate their predictions on unseen data. This
study has used a k-fold cross validation technique to assess the performance and
generalisation of the model. It involves dividing the available data into k subsets or folds, using
k-1 folds for training the model and the remaining fold for testing or validation (Li, 2023). The
process is repeated k times, with each fold serving as the validation set once. One of the key
advantages of k-fold cross-validation is that it provides a more robust estimate of model
performance compared to a single train-test split (Fushiki, 2009). By repeating the process k
times and averaging the results, it reduces the impact of the specific data partitioning on the
evaluation metrics. This helps to mitigate the risk of overfitting or underfitting that can occur
with a single train-test split.
The whole methodology summarised into a flow chart is shown in Figure 11.
31

GEOG71200 Dissertation (MSc Geographical Information Science)   Student ID: 11161619

Figure 11: Flow chart showing the methodology.
4.  Results
4.1 Feature Extraction
There are a total of 68,404 roads in the study area which are classified into primary, secondary,
tertiary, etc. as given in Table 1.
Table 1: The different classes of roads in the dataset.
| Class          | Count  | Class          | Count  |
| -------------- | ------ | -------------- | ------ |
| bridleway      | 8      | tertiary       | 2926   |
| cycleway       | 29     | tertiary_link  | 56     |
| footway        | 812    | track          | 2703   |
| living_street  | 400    | track_grade1   | 18     |
| path           | 2847   | track_grade2   | 20     |
32

GEOG71200 Dissertation (MSc Geographical Information Science)   Student ID: 11161619
|     | pedestrian  | 92  track_grade3    | 17  |
| --- | ----------- | ------------------- | --- |
|     | primary     | 1384  track_grade4  | 15  |

|     | primary_link  | 98  track_grade5  | 10  |
| --- | ------------- | ----------------- | --- |

|     | residential     | 33194  trunk      | 615    |
| --- | --------------- | ----------------- | ------ |
|     | secondary       | 1092  trunk_link  | 154    |
|     | secondary_link  | 55  unclassified  | 18342  |

|     | service  | 3390  unknown  | 52  |
| --- | -------- | -------------- | --- |

|     | steps  | 75  Total  | 68404  |
| --- | ------ | ---------- | ------ |

Out of the total number of roads, only 765 were flooded and the rest 67639 were not flooded.
The statistics of the different characteristics of each of the road is shown in Tables 2 and 3.

  Table 2: Summary statistics of min_elevation, max_elevation, min_slope,
min_precipitation and max_precipitation.
33

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Table 3: Summary statistics of mean_elevation, mean_slope, mean_precipitation,
imperviousness, and distance_to_drainage.
4.2 Logistic Regression Statistics for Feature Selection
The summary statistics from the Logistic Regression model is shown in Figure 12.
Figure 12: Summary statistics from the Logistic Regression model used
for feature selection.
34

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Upon evaluating the logistic regression model for the 'flooded' outcome, it was discerned that
the model optimally converged after 10 iterations. In terms of model fit, the pseudo R-squared
value stood at -0.079, not directly indicative of variance explained like in linear regression,
and possibly suggesting a fit no better than a model with just a constant. This is further
accentuated by its log-likelihood value of -4529.2, and an LLR p-value of 1.0000, hinting that
the model's statistical superiority over a rudimentary, intercept-only model might be
questionable.
Interpreting Significant Coefficients:
Mean slope: For every unit increment in mean_slope, the log-odds of 'flooded' decrease by
0.0298. This observation is statistically significant (p < 0.05).
Imperviousness: A unit rise in imperviousness causes the log-odds of 'flooded' to decrease by
0.0347.
Distance to drainage: A single unit increase corresponds to a reduction in the log-odds of
'flooded' by 0.0034, a statistically crucial correlation.
Minimum elevation: Each unit enhancement in min_elevation leads to a decline in the log-
odds of 'flooded' by 0.0068.
Minimum slope: An increment in min_slope reduces the log-odds of 'flooded' by 0.0506.
Maximum slope: An increase in max_slope augments the log-odds by 0.0345, showing its
significance in the model.
Potentially Insignificant Predictors:
Certain variables like mean_elevation, mean_precipitation, max_elevation,
min_precipitation, and max_precipitation present p-values exceeding the 0.05 threshold,
hinting that their significance in predicting the 'flooded' outcome might be questionable in
this model's context.
35

GEOG71200 Dissertation (MSc Geographical Information Science)   Student ID: 11161619
4.3 Recursive Feature Elimination Results
Recursive feature elimination was also used on the same database to identify the 5 most
significant features. The results after running the script is shown in Figure 13.  These results
can be interpreted as shown in Table 4.
Figure 13: Results from Recursive Feature Elimination.

Table 4: Significance of Predictors in the Flood Model.
| Predictor             | Significance     | Rank  |
| --------------------- | ---------------- | ----- |
| Mean Elevation        | Not Significant  | 5     |
| Mean Slope            | Significant      | 1     |
| Mean Precipitation    | Significant      | 1     |
| Imperviousness        | Significant      | 1     |
| Distance to Drainage  | Not Significant  | 7     |
| Min Elevation         | Not Significant  | 6     |
| Max Elevation         | Not Significant  | 4     |
| Min Slope             | Significant      | 1     |
| Max Slope             | Significant      | 1     |
| Min Precipitation     | Not Significant  | 2     |
| Max Precipitation     | Not Significant  | 3     |

36

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
The table indicates the significance and rank of each predictor in determining flood likelihood.
Predictors marked as "Significant" were found to be vital for the model, whereas those labeled
as "Not Significant" did not significantly influence the outcome based on recursive feature
extraction.
Among the predictors, 'Mean Slope', 'Mean Precipitation', 'Imperviousness', 'Min Slope', and
'Max Slope' were identified as the most influential factors in the prediction model.
4.4 Selected Features
Based on the results from the logistic regression statistics and recursive feature elimination,
the features mean slope, mean precipitation, imperviousness, minimum slope, maximum
slope, minimum elevation and distance to drainage were chosen as the features that were
used to train the model. Histograms showing the distribution of these characteristics is shown
in Figures 14 and 15.
37

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Figure 14: Histograms showing distribution of mean_slope, mean_precipitation, imperviousness
and distance_to_drainage.
Figure 15: Histograms showing distribution of min_elevation, min_slope and max_slope.
4.5 Model Performance
The dataset was then used to train the model based on Logistic Regression. The data
contained 68,404 rows with 8 columns, one of which stored binary values 1 or 0 indicating
if the road was flooded or not. But there was a large imbalance in the data with 98.88
38

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
percentage of the data being that of roads that were not flooded and with only 1.12
percentage being flooded roads. A chart showing this distribution is shown in Figure 16.
The mean value of each of the features for flooded and non-flooded roads respectively
are shown in Table 5.
To get rid of the data imbalance, the SMOTE algorithm was used. The distribution of the
data after this is exactly half of the database having roads that are flooded and the rest
non flooded. This is achieved by creating synthetic samples of the minority class which in
this case is roads that are flooded. After this, logistic regression model was trained by
39
flo o d e d
0
1
m e a n _ s lo p e
1 0 .5
1 9 .3
5
3
m e a n _ pio r en c ip
1 8
1 6
ita t
.4 6
.0 3
im p e r v io u s n e s
4 .9
2 .0
s
5
5
d is
_ d
tar
a
n c e _ to
in a g e
2 5 9 .4 7
1 3 6 .8 2
m in _ e le v
2
5
a
9
1
tio n
9 .4 6
1 .3 5
m in _ s lo
5
6
p
.4 .0
e
2
9
m a x _ s lo
1 6
3 6
p
.4 .7
e
1
7
Figure 16: Chart showing the distribution of flooded (1) and non-
flooded (0) roads in the dataset.
Table 5: The mean values of all the characteristics used for training the model.

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
taking 70 percent of this data. When the model was tested on the rest 30 percent, it gave
an accuracy of 0.74. The confusion matrix of the model’s performance is shown in Figure
17.
Figure 17: Confusion Matrix of the model’s performance.
From Table 6 which are metrics derived from the confusion matrix, it can be seen that for
Class 0, the precision is 0.71, recall is 0.81, and the F1-score stands at 0.76. In contrast, for
Class 1, precision is reported at 0.78, recall at 0.66, and the F1-score is 0.72. The overall
accuracy of the model across both classes is 74%. Both the macro and weighted averages
for precision, recall, and F1-score consistently measure at approximately 0.74.
40

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Table 6: Metrics derived from confusion matrix.
The Receiver Operating Characteristic (ROC) curve of the model’s performance is given in
Figure 18.
Figure 18: ROC curve of the model’s performance.
\
41

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
The ROC curve illustrates the performance of a logistic regression model for binary
classification. Positioned above the diagonal line, the model demonstrates better predictive
power than random guessing, with an Area Under the Curve (AUC) of 0.74, suggesting good
but not perfect discrimination between classes. This curve plots the True Positive Rate (TPR)
against the False Positive Rate (FPR) for various classification thresholds, with the trajectory
indicating the model's trade-offs between TPR and FPR. While the model's performance is
reasonably good, its adequacy depends on the specific application and context.
Finally, the model was validated using the k-fold cross validation which gave an accuracy of
0.7367 (+/- 0.0046).
5. Discussions
5.1 Vulnerability Assessment of Road Network
Elevation's role in flood vulnerability is pivotal. Typically, lower elevations seem more at risk.
In this study, min_elevation and mean_slope have emerged as noteworthy predictors.
Specifically, an increase in min_elevation suggests roads at higher elevations are generally less
prone to flooding due to the natural flow of water towards lower areas. However, in terrains
where higher elevations channel water, rapid runoff can lead to flash floods.
The slope, both min_slope and max_slope, significantly influences flooding dynamics. Roads
with a defined slope could encourage faster water runoff, minimizing water accumulation.
However, rapid runoff can result in downstream areas flooding, especially if unprepared for
such water influxes.
Imperviousness highlights the anthropogenic factor influencing flooding. Urbanization often
increases surfaces that prevent water absorption. With rising imperviousness, rainwater
struggles to find outlets, boosting the likelihood of flooding. This underscores the importance
of permeable materials and efficient drainage in urban planning.
Distance_to_drainage reflects the inherent risks associated with proximity to water bodies or
drainage systems. Roads closer to these systems might be at increased risk, especially if the
drainage systems are not adequately maintained or are overwhelmed during intense rainfall.
42

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
5.2 Creating an Algorithm to Predict Flooding and Feature Importance
The endeavour to harness the power of machine learning in flood prediction reflects a shift
from traditional modelling approaches, driven by the need for greater accuracy and
adaptability. The logistic regression model, rooted in machine learning, offers a promising
alternative by capturing the intricate interplays between various factors.
The initial data imbalance, where a majority of instances represented non-flooded roads, was
a significant challenge. In machine learning, a skewed dataset can result in models that are
biased towards the majority class. Such models might achieve high accuracy by merely
predicting the majority class but fail to capture the nuances of the minority class, which in
this case is critical.
Incorporating the SMOTE algorithm was a strategic decision. By generating synthetic instances
of the minority class, the algorithm enriched the dataset, allowing for a more balanced
training process. This ensures the model is not just statistically sound but also practically
relevant.
Furthermore, the application of Recursive Feature Extraction (RFE) provided illuminating
insights into feature importance. The results from RFE showcased five significant features:
mean_slope, mean_precipitation, imperviousness, min_slope, and max_slope. It's worth
noting that while other variables might have intrinsic importance, these five stood out in their
predictive capability. This suggests that the slopes (both mean and minimum) and
imperviousness play central roles in determining flood vulnerability, reinforcing the earlier
discussions. Meanwhile, mean_precipitation emphasizes the direct role of rainfall intensity in
flooding events.
5.3 Validating the Algorithm with Data from the 2018 Floods
The catastrophic 2018 Kerala floods offered a real-world context to test the model's
robustness. Given the widespread devastation, particularly to the road networks, any model's
validation against this backdrop would be stringent.
The model's performance metrics, derived from the confusion matrix, indicated an overall
accuracy of 0.74 post-SMOTE. While this signifies the model's robust predictive capabilities,
43

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
focusing on precision for both Class 0 and Class 1 (0.71 and 0.78 respectively) demonstrates
the model's ability to correctly identify the true positive rate, while recall for these classes
(0.81 and 0.66 respectively) showcases the model's ability to capture actual positive
instances. The F1-score, a harmonic mean of precision and recall, further affirms the model's
balanced accuracy for both classes.
Further validation through the k-fold cross-validation yielded a consistent accuracy of 0.7367
with a minor variability of +/- 0.0046, adding another layer of confidence to the model's
reliability and stability.
The ROC curve's analysis further complements these insights, showcasing the trade-off
between sensitivity (true positive rate) and specificity (true negative rate). The closer the
curve is to the top-left corner, the better the model's performance.
While the algorithm has showcased commendable predictive prowess, it's imperative to
remember that models, however advanced, have limitations. External factors, data
anomalies, or unprecedented climate events can always introduce uncertainties.
6. Conclusions
The pressing global concern of increased flooding events, driven by climate change and
human-induced changes to landscapes, underscores the imperative need for effective,
predictive measures, especially where urban infrastructure, such as roads, is concerned. The
significant aftermath of the 2018 Kerala floods, especially in the Ernakulam and Idukki
districts, accentuated this urgency, bringing to light the crippling effects of such catastrophic
events on transportation networks and, by extension, on communities and economies.
This study aimed to address the prevalent issue of road network disruption during heavy
precipitation events. It ambitiously sought to harness machine learning techniques, notably
logistic regression, to predict road disruptions, grounding its research in the severe 2018
floods in Kerala. As delineated in the methodology, the approach was both systematic and
holistic, spanning a gamut of steps from comprehensive data acquisition to intricate feature
extraction and rectification of data imbalance, before culminating in model training and
validation.
44

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
The raw data, predominantly sourced from remote sensing products due to scarcity in
available data post the 2018 floods, provided a foundation for extracting features pertinent
to road networks, such as elevation, slope, permeability, and distance to drainage, among
others. Importantly, the data imbalance—a prevalent issue in many natural disaster
datasets—was meticulously addressed using the SMOTE method, ensuring that the predictive
model would not be overly skewed towards the majority class and would provide a more
holistic picture.
The logistic regression model's choice, while rooted in its computational efficiency and
capability to provide insights into the relative importance of input variables, also underscored
a significant point—the potential of traditional statistical techniques when applied
thoughtfully to modern-day challenges. Although the model's pseudo R-squared value and
other statistics suggested potential areas of improvement, its insights into significant features
were invaluable.
For instance, the importance of features like mean_slope, imperviousness, and
distance_to_drainage reinforces the necessity of considering geographical and topographical
factors in urban planning and flood mitigation. These features not only affect immediate flood
responses but also have long-term impacts on a region's resilience to such events.
Furthermore, the recursive feature extraction provided a refined perspective on the
significant predictors, validating some of our initial assumptions while offering newer insights.
This iterative process of feature selection underscored the importance of leveraging both
statistical and machine learning techniques to refine our predictive capabilities.
However, the broader conclusion from this research transcends its technical achievements.
At its core, this study reinforces the importance of proactive, data-driven decision-making in
urban planning and disaster response. The ability to predict, with reasonable accuracy, which
roads might be disrupted during heavy rainfall can empower local authorities to act in
advance, be it in rerouting traffic, reinforcing vulnerable infrastructure, or deploying
emergency response teams more effectively.
Furthermore, this study underscores the importance of interdisciplinary collaboration—
bridging the gap between traditional hydrological insights, geospatial analysis, and modern
45

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
data science techniques. In a rapidly changing world, with the spectre of climate change
making such events more commonplace, such collaborations are not just beneficial but
essential.
Reflecting on the personal motivation behind this study, which emerges from witnessing the
devastation of the 2018 Kerala floods firsthand, it's evident that the stakes are high. Ensuring
their resilience ensures that a community can withstand, recover, and thrive post-disasters.
While this study provides a blueprint for predicting road disruptions during heavy rainfall in
the specific context of Kerala, its methodology and insights can be adapted and applied to
other regions, making it a valuable contribution to the broader discourse on urban resilience
and flood preparedness. The hope is that the findings of this research will serve as a catalyst
for more such predictive initiatives, underpinning global efforts to fortify urban landscapes
against the unpredictable wrath of nature.
46

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
7. References
Abshirini, E. and Koch, D. (2016). ‘Rivers as integration devices in cities’. City, Territory and
Architecture, 3(1). https://doi.org/10.1186/s40410-016-0030-4
Ananthakrishnan, R. and Soman, M. K. (1988). ‘The onset of the southwest monsoon over Kerala:
1901-1980’. Journal of Climatology, 8(3), pp. 283-296. https://doi.org/10.1002/joc.3370080305
Andreadis, K.M., Wing, O.E., Colven, E., Gleason, C.J., Bates, P.D. and Brown, C.M., (2022).
‘Urbanizing the floodplain: global changes of imperviousness in flood-prone areas’.
Environmental Research Letters, 17(10), p.104024.
Antonetti, M., Horat, C., Sideris, I. V., & Zappa, M. (2019). ‘Ensemble flood forecasting considering
dominant runoff processes – part 1: set-up and application to nested basins (Emme, Switzerland)’.
Natural Hazards and Earth System Sciences, 19(1), pp. 19-40. https://doi.org/10.5194/nhess-19-
19-2019
Asim, M., Mekkodathil, A., Sathian, B., Elayedath, R., N, R. K., Simkhada, P., … & Teijlingen, E. v.
(2019). ‘Post-traumatic stress disorder among the flood affected population in Indian
subcontinent’. Nepal Journal of Epidemiology, 9(1), pp. 755-758.
https://doi.org/10.3126/nje.v9i1.24003
Atashi, V., Gorji, H. T., Shahabi, S. M., Kardan, R., & Lim, Y. H. (2022). ‘Water level forecasting using
deep learning time-series analysis: a case study of red river of the north’. Water, 14(12), p.1971.
https://doi.org/10.3390/w14121971
Babu, S. M. and Leno, N. (2020). ‘Organic carbon status in flood affected high hills of Kerala’.
International Journal of Chemical Studies, 8(4), pp. 3033-3035.
https://doi.org/10.22271/chemi.2020.v8.i4ak.10111
Bach, M., Werner, A., ?ywiec, J., & Pluskiewicz, W. (2017). ‘The study of under- and over-sampling
methods’ utility in analysis of highly imbalanced data on osteoporosis’. Information Sciences, 384,
pp. 174-190. https://doi.org/10.1016/j.ins.2016.09.038
Baldassarre, G. D., Montanari, A., Lins, H. F., Koutsoyiannis, D., Brandimarte, L., & Blöschl, G.
(2010). ‘Flood fatalities in africa: from diagnosis to mitigation’. Geophysical Research Letters,
37(22). https://doi.org/10.1029/2010gl045467
Banihabib, M. E., Jurík, ?., Kazemi, M. S., Soltani, J., & Tanhapour, M. (2020). ‘A hybrid intelligence
model for the prediction of the peak flow of debris floods’. Water, 12(8), p. 2246.
https://doi.org/10.3390/w12082246
Bayissa, Y. A., Tadesse, T., Demisse, G. B., & Shiferaw, A. (2017). ‘Evaluation of satellite-based
rainfall estimates and application to monitor meteorological drought for the upper blue Nile
basin, Ethiopia’. Remote Sensing, 9(7), p. 669. https://doi.org/10.3390/rs9070669
Berkhahn, S., Fuchs, L., & Neuweiler, I. (2019). ‘An ensemble neural network model for real-time
prediction of urban floods’. Journal of Hydrology, 575, pp. 743-754.
https://doi.org/10.1016/j.jhydrol.2019.05.066
47

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Brockkötter, J., Ahndorf, J., & Jupke, A. (2021). ‘Prediction of flooding in packed liquid-liquid and
high-pressure extraction columns using a gaussian process’. Chemie Ingenieur Technik, 93(12), pp.
1907-1916. https://doi.org/10.1002/cite.202100073
Ceola, S., Laio, F., & Montanari, A. (2014). ‘Satellite nighttime lights reveal increasing human
exposure to floods worldwide’. Geophysical Research Letters, 41(20), pp. 7184-7190.
https://doi.org/10.1002/2014gl061859
Chang, D., Yang, S., Hsieh, S. L., Wang, H. J., & Yeh, K. (2020). ‘Artificial intelligence methodologies
applied to prompt pluvial flood estimation and prediction’. Water, 12(12), p. 3552.
https://doi.org/10.3390/w12123552
Chen, C., Luan, D., Zhao, S., Liao, Z., Zhou, Y., Jiang, J., … & Pei, Q. (2021). ‘Flood discharge
prediction based on remote-sensed spatiotemporal features fusion and graph attention’. Remote
Sensing, 13(24), p. 5023. https://doi.org/10.3390/rs13245023
Chen, Y., Zhang, X., Yang, K., Zeng, S., & Hong, A. (2023). ‘Modeling rules of regional flash flood
susceptibility prediction using different machine learning models’. Frontiers in Earth Science, 11.
https://doi.org/10.3389/feart.2023.1117004
Choubin, B., Moradi, E., Golshan, M., Adamowski, J., Hosseini, F. S., & Mosavi, A. (2019). ‘An
ensemble prediction of flood susceptibility using multivariate discriminant analysis, classification
and regression trees, and support vector machines’. Science of the Total Environment, 651, pp.
2087-2096. https://doi.org/10.1016/j.scitotenv.2018.10.064
Costache, R. and Bui, D. T. (2020). ‘Identification of areas prone to flash-flood phenomena using
multiple-criteria decision-making, bivariate statistics, machine learning and their ensembles’.
Science of the Total Environment, 712, p. 136492.
https://doi.org/10.1016/j.scitotenv.2019.136492
Dang, A. T. N. and Kumar, L. (2017). ‘Application of remote sensing and GIS-based hydrological
modelling for flood risk analysis: a case study of district 8, ho chi minh city, Vietnam’. Geomatics,
Natural Hazards and Risk, 8(2), pp. 1792-1811. https://doi.org/10.1080/19475705.2017.1388853
Danso, S. Y., Yi, M., Adjakloe, Y. A., & Addo, I. Y. (2020). ‘Application of an index-based approach
in geospatial techniques for the mapping of flood hazard areas: a case of cape coast metropolis
in Ghana’. Water, 12(12), p. 3483. https://doi.org/10.3390/w12123483
Dhasmana, M. K., Mondal, A., & Zachariah, M. (2023). ‘On the role of climate change in the 2018
flooding event in Kerala’. Environmental Research Letters, 18(8), p. 084016.
https://doi.org/10.1088/1748-9326/ace6c0
Dinku, T., Funk, C., Peterson, P. Y., Maidment, R., Tadesse, T., Gadain, H., … & Ceccato, P. (2018).
‘Validation of the chirps satellite rainfall estimates over eastern Africa’. Quarterly Journal of the
Royal Meteorological Society, 144(S1), pp. 292-312. https://doi.org/10.1002/qj.3244
Divakaran, S. J., Philip, J. S., Chereddy, P., Nori, S. R. C., Ganesh, A. J., John, J., … & Nelson-Sathi, S.
(2019). ‘Insights into the bacterial profiles and resistome structures following the severe 2018
48

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
flood in Kerala, south India’. Microorganisms, 7(10), p. 474.
https://doi.org/10.3390/microorganisms7100474
Dixit, A., Sahany, S., Rajagopalan, B., & Choubey, S. (2022). ,Role of changing land use and land
cover (lulc) on the 2018 megafloods over Kerala, India,. Climate Research, 89, pp. 1-14.
https://doi.org/10.3354/cr01701
Elhanafy, H. and Copeland, G. J. M. (2007). ‘Flash floods simulation using saint venant equations’.
International Conference on Aerospace Sciences and Aviation Technology, 12(ASAT CONFERENCE),
pp. 1-14. https://doi.org/10.21608/asat.2007.23886
Esfandiari, M., Abdi, G., Jabari, S., McGrath, H., & Coleman, D. (2020). ‘Flood hazard risk mapping
using a pseudo supervised random forest’. Remote Sensing, 12(19), p. 3206.
https://doi.org/10.3390/rs12193206
Farr, T. G., Rosen, P. A., Caro, E., Crippen, R. E., Duren, R., Hensley, S., … & Alsdorf, D. (2007). ‘The
shuttle radar topography mission’. Reviews of Geophysics, 45(2).
https://doi.org/10.1029/2005rg000183
Ferencevic, M. V. and Ashmore, P. (2011). ‘Creating and evaluating digital elevation model-based
stream-power map as a stream assessment tool’. River Research and Applications, 28(9), pp.
1394-1416. https://doi.org/10.1002/rra.1523
Frederikse, T., Lee, T., Wang, O., Kirtman, B., Becker, E., Hamlington, B. D., … & Waliser, D. E.
(2022). ‘A hybrid dynamical approach for seasonal prediction of sea-level anomalies: a pilot study
for Charleston, south Carolina’. Journal of Geophysical Research: Oceans, 127(8).
https://doi.org/10.1029/2021jc018137
Funk, C., Peterson, P. Y., Landsfeld, M. F., Pedreros, D., Verdin, J. P., Shukla, S., … & Michaelsen, J.
(2015). ‘The climate hazards infrared precipitation with stations—a new environmental record for
monitoring extremes’. Scientific Data, 2(1). https://doi.org/10.1038/sdata.2015.66
Fushiki, T. (2009). ‘Estimation of prediction error by using k-fold cross-validation’. Statistics and
Computing, 21(2), pp. 137-146. https://doi.org/10.1007/s11222-009-9153-8
George, R., Joseph, A., & Rema, K. P. (2022). ‘Application of Hydrologic Engineering Centers
Hydrologic Modeling System (hec-hms) model for modelling flood in sub basin of Meenachil river,
Kerala, India’. International Journal of Environment and Climate Change, pp. 1251-1262.
https://doi.org/10.9734/ijecc/2022/v12i121564
Ginkel, K. v., Dottori, F., Alfieri, L., Feyen, L., & Koks, E. (2021). ‘Flood risk assessment of the
european road network’. Natural Hazards and Earth System Sciences, 21(3), pp. 1011-1027.
https://doi.org/10.5194/nhess-21-1011-2021
Giovannettone, J., Copenhaver, T., Burns, M., & Choquette, S. R. (2018). ‘A statistical approach to
mapping flood susceptibility in the lower Connecticut river valley region’. Water Resources
Research, 54(10), pp. 7603-7618. https://doi.org/10.1029/2018wr023018
49

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Gong, P., Li, X., Wang, J., Bai, Y., Chen, B., Hu, T., ... & Zhou, Y. (2020). ‘Annual maps of global
artificial impervious area (GAIA) between 1985 and 2018’. Remote Sensing of Environment, 236,
p. 111510.
Granata, F., Gargano, R., & Marinis, G. d. (2016). ‘Support vector regression for rainfall-runoff
modeling in urban drainage: a comparison with the epa’s storm water management model’.
Water, 8(3), p. 69. https://doi.org/10.3390/w8030069
Greeshma, S. and Greeshma, M. (2023). ‘The economic impact of floods on the plantation sector:
a study of selected districts in Kerala’. Disaster Advances, 16(3), pp. 13-22.
https://doi.org/10.25303/1603da013022
Guan, M., Liang, Q., & Hou, J. (2021). ‘Editorial: smart approaches to predict urban flooding:
current advances and challenges’. Frontiers in Earth Science, 9.
https://doi.org/10.3389/feart.2021.681751
Ha, H., Luu, C., Bui, Q. D., Pham, D., Hoang, T., Nguyen, V., … & Pham, B. T. (2021). ‘Flash flood
susceptibility prediction mapping for a road network using hybrid machine learning models’.
Natural Hazards, 109(1), pp. 1247-1270. https://doi.org/10.1007/s11069-021-04877-5
Haklay, M. and Weber, P. (2008). ‘Openstreetmap: user-generated street maps’. IEEE Pervasive
Computing, 7(4), 12-18. https://doi.org/10.1109/mprv.2008.80
Hashemi-Beni, L., Jones, J., Thompson, G. L., Johnson, C., & Gebrehiwot, A. (2018). ‘Challenges
and opportunities for uav-based digital elevation model generation for flood-risk management: a
case of Princeville, North Carolina’. Sensors, 18(11), p. 3843. https://doi.org/10.3390/s18113843
Hettiarachchi, S., Wasko, C., & Sharma, A. (2018). ‘Increase in flood risk resulting from climate
change in a developed urban watershed – the role of storm temporal patterns’. Hydrology and
Earth System Sciences, 22(3), pp. 2041-2056. https://doi.org/10.5194/hess-22-2041-2018
Hinkel, J., Lincke, D., Vafeidis, A. T., Perrette, M., Nicholls, R. J., Tol, R. S., … & Levermann, A. (2014).
‘Coastal flood damage and adaptation costs under 21st century sea-level rise’. Proceedings of the
National Academy of Sciences, 111(9), pp. 3292-3297. https://doi.org/10.1073/pnas.1222469111
Hou, L., Wang, Y., Shen, F., Lei, M., Wang, X., Zhao, X., … & Alhaj, A. (2020). ‘Study on variation of
surface runoff and soil moisture content in the subgrade of permeable pavement structure’.
Advances in Civil Engineering, 2020, pp. 1-12. https://doi.org/10.1155/2020/8836643
Hu, Q., Zhu, Y., Hu, H., Guan, Z., Qian, Z., & Yang, A. (2021). ‘Multiple kernel learning with
maximum inundation extent from modis imagery for spatial prediction of flood susceptibility’.
https://doi.org/10.21203/rs.3.rs-685721/v1
Huang, M. and Jin, S. (2020). ‘Rapid flood mapping and evaluation with a supervised classifier and
change detection in shouguang using sentinel-1 sar and sentinel-2 optical data. Remote Sensing,
12(13), p. 2073. https://doi.org/10.3390/rs12132073
Hunt, K. M. R. and Menon, A. (2020). ‘The 2018 Kerala floods: a climate change perspective’.
Climate Dynamics, 54(3-4), pp. 2433-2446. https://doi.org/10.1007/s00382-020-05123-7
50

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Huong, H. T. L. and Pathirana, A. (2013). ‘Urbanization and climate change impacts on future
urban flooding in can tho city, Vietnam’. Hydrology and Earth System Sciences, 17(1), pp. 379-394.
https://doi.org/10.5194/hess-17-379-2013
Ishak, E. H., Haddad, K., Zaman, M. A., & Rahman, A. (2011). ‘Scaling property of regional floods
in new south Wales Australia’. Natural Hazards, 58(3), pp. 1155-1167.
https://doi.org/10.1007/s11069-011-9719-6
Jafarzadegan, K., Moradkhani, H., Pappenberger, F., Moftakhari, H., Bates, P., Abbaszadeh, P., … &
Duan, Q. (2023). ‘Recent advances and new frontiers in riverine and coastal flood modeling’.
Reviews of Geophysics, 61(2). https://doi.org/10.1029/2022rg000788
Jia, J., Cui, W., & Liu, J. (2022). ‘Urban catchment-scale blue-green-gray infrastructure
classification with unmanned aerial vehicle images and machine learning algorithms’. Frontiers in
Environmental Science, 9. https://doi.org/10.3389/fenvs.2021.778598
Jose, M. and Fenn, J. (2021). ‘Post-traumatic stress disorder and resilience among flood affected
farmers of Kerala, India’. International Journal of Community Medicine and Public Health, 8(4), p.
1842. https://doi.org/10.18203/2394-6040.ijcmph20211243
Joseph, J. E. (2022). ‘Social media response in disaster management with special reference to the
Kerala floods’. Society and Culture Development in India, 2(2), pp. 291-303.
https://doi.org/10.47509/scdi.2022.v02i02.04
Khalaf, M., Alaskar, H., Hussain, A. J., Baker, T., Maamar, Z., Buyya, R., … & Al-Jumeily, D. (2020).
‘Iot-enabled flood severity prediction via ensemble machine learning models’. IEEE Access, 8, pp.
70375-70386. https://doi.org/10.1109/access.2020.2986090
Kim, B., Yum, S., Kim, Y. Y., Yun, N., Shin, S., & You, S. (2014). ‘An analysis of factors relating to
agricultural machinery farm-work accidents using logistic regression’. Journal of Biosystems
Engineering, 39(3), pp. 151-157. https://doi.org/10.5307/jbe.2014.39.3.151
Kohansarbaz, A., Shabanlou, S., Yosefvand, F., & Rajabi, A. (2022). ‘Modelling flood susceptibility
in northern Iran: application of five well-known machine-learning models’. Irrigation and
Drainage, 71(5), pp. 1332-1350. https://doi.org/10.1002/ird.2745
Kondo, R., Du, B., Narusue, Y., & Morikawa, H. (2023). ‘Machine learning framework supervised
by hydraulic mechanical models for real-time pluvial flood prediction’. Journal of Information
Processing, 31(0), pp. 256-264. https://doi.org/10.2197/ipsjjip.31.256
Kong, J., Simonovi?,? S. P., & Zhang, C. (2019). ‘Resilience assessment of interdependent
infrastructure systems: a case study based on different response strategies’. Sustainability, 11(23),
p. 6552. https://doi.org/10.3390/su11236552
Kumar, V., Pradhan, P. K., Sinha, T., Rao, S. V. B., & Chang, H. (2020). ‘Interaction of a low-pressure
system, an offshore trough, and mid-tropospheric dry air intrusion: the Kerala flood of august
2018’. Atmosphere, 11(7), p. 740. https://doi.org/10.3390/atmos11070740
51

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Kundzewicz, Z. W., Kanae, S., Seneviratne, S. I., Handmer, J., Nicholls, N., Peduzzi, P., … &
Sherstyukov, B. G. (2013). ‘Flood risk and climate change: global and regional perspectives’.
Hydrological Sciences Journal, 59(1), pp. 1-28. https://doi.org/10.1080/02626667.2013.857411
Lamb, R. A., Garside, P., Pant, R., & Hall, J. W. (2019). ‘A probabilistic model of the economic risk
to britain's railway network from bridge scour during floods’. Risk Analysis, 39(11), pp. 2457-2478.
https://doi.org/10.1111/risa.13370
Lendering, K., Jonkman, S. N., & Kok, M. (2015). ‘Effectiveness of emergency measures for flood
prevention’. Journal of Flood Risk Management, 9(4), pp. 320-334.
https://doi.org/10.1111/jfr3.12185
Lhomme, J., Sayers, P., Gouldby, B., Samuels, P., Wills, M., & Mulet-Marti, J. (2008). ‘Recent
development and application of a rapid flood spreading method’. Flood Risk Management:
Research and Practice, pp. 15-24. https://doi.org/10.1201/9780203883020.ch2
Li, B., Gao, Y., Wan, R., Dai, X., & Zhang, Y. (2016). ‘Comparison of random forests and other
statistical methods for the prediction of lake water level: a case study of the Poyang lake in China’.
Hydrology Research, 47(S1), pp. 69-83. https://doi.org/10.2166/nh.2016.264
Li, X., Yan, D., Wang, K., Weng, B., Qin, T., & Liu, S. (2019). ‘Flood risk assessment of global
watersheds based on multiple machine learning models’. Water, 11(8), p. 1654.
https://doi.org/10.3390/w11081654
Li, Z., Liu, H., Luo, C., & Fu, G. (2021). ‘Assessing surface water flood risks in urban areas using
machine learning’. Water, 13(24), pp. 3520. https://doi.org/10.3390/w13243520
Li, Z., Wu, J., Yang, J., Li, K., Chen, J., Huang, S., … & Chen, P. (2023). ‘Genome-wide association
studies combined with k-fold cross-validation identify rs17822931 as an ancestry-informative
marker in Han Chinese population’. Electrophoresis, 44(15-16), pp. 1187-1196.
https://doi.org/10.1002/elps.202200227
Liu, J., Wang, J., Xiong, J., Cheng, W., Sun, H., Yong, Z., … & Wang, N. (2021). ‘Hybrid models
incorporating bivariate statistics and machine learning methods for flash flood susceptibility
assessment based on remote sensing datasets’. Remote Sensing, 13(23), p. 4945.
https://doi.org/10.3390/rs13234945
Lü, X., Zhong, M., & Zha, D. (2022). ‘Runoff forecasting using machine-learning methods: case
study in the middle reaches of xijiang river’. Frontiers in Big Data, 4.
https://doi.org/10.3389/fdata.2021.752406
Madsen, H., Lawrence, D., Lang, M., Martínková, M., & Kjeldsen, T. (2014). ‘Review of trend
analysis and climate change projections of extreme precipitation and floods in Europe’. Journal of
Hydrology, 519, pp. 3634-3650. https://doi.org/10.1016/j.jhydrol.2014.11.003
Mah, D. Y. S., Hin, L. S., & Putuhena, F. J. (2012). ‘Investigative modelling of the flood bypass
channel in kuching, sarawak, by assessing its impact on the inundations of kuching-batu kawa-
bau expressway’. Structure and Infrastructure Engineering, 8(7), pp. 705-714.
https://doi.org/10.1080/15732471003770167
52

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Mansour, M. (2020). ‘A participatory approach for identification of micro flood zones in poorly
developed urban areas’. Academic Perspective Procedia, 3(2), pp. 941-949.
https://doi.org/10.33793/acperpro.03.02.32
Mishra, V. and Shah, H. L. (2018).’ Hydroclimatological perspective of the Kerala flood of 2018’.
Journal of the Geological Society of India, 92(5), pp. 645-650. https://doi.org/10.1007/s12594-
018-1079-3
Mishra, V., Aaadhar, S., Shah, H. L., Kumar, R., Pattanaik, D. R., & Tiwari, A. D. (2018). ‘The Kerala
flood of 2018: combined impact of extreme rainfall and reservoir storage’.
https://doi.org/10.5194/hess-2018-480
Mistry, S. and Parekh, F. (2022). ‘Flood forecasting using artificial neural network’. IOP Conference
Series: Earth and Environmental Science, 1086(1), p. 012036. https://doi.org/10.1088/1755-
1315/1086/1/012036
Mosavi, A., Öztürk, P., & Chau, K. W. (2018). ‘Flood prediction using machine learning models:
literature review’. Water, 10(11), p. 1536. https://doi.org/10.3390/w10111536
Munawar, H. S., Hammad, A. W. A., & Waller, S. T. (2022). ‘Remote sensing methods for flood
prediction: a review’. Sensors, 22(3), p. 960. https://doi.org/10.3390/s22030960
Nasiri, H., Yusof, M. J. M., & Ali, T. A. M. (2016). ‘An overview to flood vulnerability assessment
methods’. Sustainable Water Resources Management, 2(3), pp. 331-336.
https://doi.org/10.1007/s40899-016-0051-x
Neeraj, S., Mannakkara, S., & Wilkinson, S. (2020). ‘Build back better concepts for resilient
recovery: a case study of india’s 2018 flood recovery’. International Journal of Disaster Resilience
in the Built Environment, 12(3), pp. 280-294. https://doi.org/10.1108/ijdrbe-05-2020-0044
Pant, R., Thacker, S., Alderson, D., & Barr, S. (2017). ‘Critical infrastructure impact assessment due
to flood exposure’. Journal of Flood Risk Management, 11(1), pp. 22-33.
https://doi.org/10.1111/jfr3.12288
Pham, B. T., Bui, D. T., Prakash, I., & Dholakia, M. B. (2017). ‘Hybrid integration of multilayer
perceptron neural networks and machine learning ensembles for landslide susceptibility
assessment at Himalayan area (India) using GIS’. Catena, 149, pp. 52-63.
https://doi.org/10.1016/j.catena.2016.09.007
Qian, K., Mohamed, A., & Claudel, C. (2019). ‘Physics informed data driven model for flood
prediction: application of deep learning in prediction of urban flood development’.
https://doi.org/10.48550/arxiv.1908.10312
Rahman, M. M. and Davis, D. N. (2013). ‘Addressing the class imbalance problem in medical
datasets’. International Journal of Machine Learning and Computing, pp. 224-228.
https://doi.org/10.7763/ijmlc.2013.v3.307
Rahman, M. T. A., Chen, N., Elbeltagi, A., Islam, M., Alam, M., Pourghasemi, H. R., … & Dewan, A.
(2021). ‘Application of stacking hybrid machine learning algorithms in delineating multi-type
53

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
flooding in Bangladesh’. Journal of Environmental Management, 295, p. 113086.
https://doi.org/10.1016/j.jenvman.2021.113086
Rahmati, O., Darabi, H., Panahi, M., Kalantari, Z., Naghibi, S. A., Ferreira, C. S. S., … & Haghighi, A.
T. (2020). ‘Development of novel hybridized models for urban flood susceptibility mapping’.
Scientific Reports, 10(1). https://doi.org/10.1038/s41598-020-69703-7
Ramanarayanan, V. and Rajeev, K. (2020). ‘Sociodemographic profile of tobacco use and its
predictors in Kerala, India’. Population Medicine, 2(November), pp. 1-6.
https://doi.org/10.18332/popmed/128324
Ramos, H. M. and Besharat, M. (2021). ‘Urban flood risk and economic viability analyses of a
smart sustainable drainage system’. Sustainability, 13(24), p. 13889.
https://doi.org/10.3390/su132413889
Read, L. and Vogel, R. M. (2015). ‘Reliability, return periods, and risk under nonstationarity’. Water
Resources Research, 51(8), pp. 6381-6398. https://doi.org/10.1002/2015wr017089
Rentschler, J., Avner, P., Marconcini, M., Su, R., Strano, E., Vousdoukas, M. and Hallegatte, S., 2023.
‘Global evidence of rapid urban growth in flood zones since 1985’. Nature, 622(7981), pp. 87-92.
Sampurno, J., Vallaeys, V., Ardianto, R., & Hanert, E. (2022). ‘Integrated hydrodynamic and
machine learning models for compound flooding prediction in a data-scarce estuarine delta’.
https://doi.org/10.5194/npg-2021-36
Sankaranarayanan, S., Prabhakar, M., Satish, S., Jain, P., Ramprasad, A., & Krishnan, A. (2019).
‘Flood prediction based on weather parameters using deep learning’. Journal of Water and
Climate Change, 11(4), pp. 1766-1783. https://doi.org/10.2166/wcc.2019.321
Sarkar, D. and Mondal, P. (2019). ‘Flood vulnerability mapping using frequency ratio (fr) model: a
case study on kulik river basin, indo-bangladesh barind region’. Applied Water Science, 10(1).
https://doi.org/10.1007/s13201-019-1102-x
Schumann, G., Bates, P., Horritt, M. S., Matgen, P., & Pappenberger, F. (2009). ‘Progress in
integration of remote sensing–derived flood extent and stage data and hydraulic models’. Reviews
of Geophysics, 47(4). https://doi.org/10.1029/2008rg000274
Senan, C. P. C., Ajin, R. S., Danumah, J. H., Costache, R., Arabameri, A., Rajaneesh, A., … &
Kuriakose, S. L. (2022). ‘Flood vulnerability of a few areas in the foothills of the western ghats: a
comparison of ahp and f-ahp models’. Stochastic Environmental Research and Risk Assessment,
37(2), pp. 527-556. https://doi.org/10.1007/s00477-022-02267-2
Shabou, S., Ruin, I., Lutoff, C., Debionne, S., Anquetin, S., Creutin, J., … & Beaufils, X. (2017).
‘Mobrisk: a model for assessing the exposure of road users to flash flood events’. Natural Hazards
and Earth System Sciences, 17(9), pp. 1631-1651. https://doi.org/10.5194/nhess-17-1631-2017
Shankar, M. A. and Bindu, C. (2021). ‘Appraising the need for disaster mitigation in existing
planning documents of municipal corporations of Kerala in the event of past disasters’. IOP
54

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Conference Series: Materials Science and Engineering, 1114(1), p. 012039.
https://doi.org/10.1088/1757-899x/1114/1/012039
Shi, Z., Hon, J., Cheng, C., Chiang, H., & Huang, H. (2022). ‘Applying machine learning techniques
to the audit of antimicrobial prophylaxis’. Applied Sciences, 12(5), p. 2586.
https://doi.org/10.3390/app12052586
Shiu, C., Liu, S. C., Fu, C., Dai, A., & Sun, Y. (2012). ‘How much do precipitation extremes change
in a warming climate?’. Geophysical Research Letters, 39(17).
https://doi.org/10.1029/2012gl052762
Simon, A. and Mohankumar, K. (2004). ‘Spatial variability and rainfall characteristics of Kerala’.
Journal of Earth System Science, 113(2), pp. 211-221. https://doi.org/10.1007/bf02709788
Singh, P. K., Sinha, V. S. P., Vijhani, A., & Pahuja, N. (2018). ‘Vulnerability assessment of urban road
network from urban flood’. International Journal of Disaster Risk Reduction, 28, pp. 237-250.
https://doi.org/10.1016/j.ijdrr.2018.03.017
Song, K., Kim, M., Kang, H., Ham, E., Noh, J., Khim, J. S., … & Chon, J. (2022). ‘Stormwater runoff
reduction simulation model for urban flood restoration in coastal area’. Natural Hazards, 114(3),
pp. 2509-2526. https://doi.org/10.1007/s11069-022-05477-7
Tabarestani, E. S. and Afzalimehr, H. (2021). ‘Artificial neural network and multi-criteria decision-
making models for flood simulation in gis: mazandaran province, iran’. Stochastic Environmental
Research and Risk Assessment, 35(12), pp. 2439-2457. https://doi.org/10.1007/s00477-021-
01997-z
Tabbussum, R. and Dar, A. Q. (2021). ‘Performance evaluation of artificial intelligence
paradigms—artificial neural networks, fuzzy logic, and adaptive neuro-fuzzy inference system for
flood prediction’. Environmental Science and Pollution Research, 28(20), pp. 25265-25282.
https://doi.org/10.1007/s11356-021-12410-1
Tamiru, H. and Wagari, M. (2021). ‘Machine-learning and hec-ras integrated models for flood
inundation mapping in baro river basin, Ethiopia’. Modeling Earth Systems and Environment, 8(2),
pp. 2291-2303. https://doi.org/10.1007/s40808-021-01175-8
Tayfur, G., Singh, V. P., Moramarco, T., & Barbetta, S. (2018). ‘Flood hydrograph prediction using
machine learning methods’. Water, 10(8), 968. https://doi.org/10.3390/w10080968
Tehrany, M. S., Pradhan, B., & Jebur, M. N. (2014). ‘Flood susceptibility mapping using a novel
ensemble weights-of-evidence and support vector machine models in gis’. Journal of Hydrology,
512, pp. 332-343. https://doi.org/10.1016/j.jhydrol.2014.03.008
Tehrany, M. S., Pradhan, B., & Jebur, M. N. (2015). ‘Flood susceptibility analysis and its verification
using a novel ensemble support vector machine and frequency ratio method’. Stochastic
Environmental Research and Risk Assessment, 29(4), pp. 1149-1165.
https://doi.org/10.1007/s00477-015-1021-9
55

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Tellman, B., Sullivan, J. A., Kuhn, C., Kettner, A. J., Doyle, C., Brakenridge, G. R., … & Slayback, D.
A. (2021). ‘Satellite imaging reveals increased proportion of population exposed to floods’.
Nature, 596(7870), pp. 80-86. https://doi.org/10.1038/s41586-021-03695-w
Thakur, P. K., Ranjan, R., Singh, S., Dhote, P. R., Sharma, V., Srivastav, V., … & Chouksey, A. (2020).
‘Synergistic use of remote sensing, gis and hydrological models for study of august 2018 kerala
floods’. The International Archives of the Photogrammetry, Remote Sensing and Spatial
Information Sciences, XLIII-B3-2020, pp. 1263-1270. https://doi.org/10.5194/isprs-archives-xliii-
b3-2020-1263-2020
Trenberth, K. E. (2011). ‘Changes in precipitation with climate change’. Climate Research, 47(1),
pp. 123-138. https://doi.org/10.3354/cr00953
Troy, T. J., Pavao-Zuckerman, M., & Evans, T. (2015). ‘Debates—perspectives on socio-hydrology:
socio-hydrologic modeling: tradeoffs, hypothesis testing, and validation’. Water Resources
Research, 51(6), pp. 4806-4814. https://doi.org/10.1002/2015wr017046
V.L., G. (2021). ‘The impact of flood in 2018 on the socio-economic conditions of the dairy farmers
in pariyaram panchayat of Thrissur district of Kerala’. Journal of Veterinary and Animal Sciences,
52(3). https://doi.org/10.51966/jvas.2021.52.3.272-276
Varughese, A. and Purushothaman, C. (2021). ‘Climate change and public health in india: the 2018
Kerala floods’. World Medical &Amp; Health Policy, 13(1), 16-35.
https://doi.org/10.1002/wmh3.429
Veerakumaran, G. and Santhi, S. (2019). ‘Impact assessment of Kerala flood 2018 on agriculture
of farmers in Edathua panchayat, Kuttanad taluk of Alappuzha district’. Shanlax International
Journal of Economics, 7(4), pp. 24-28. https://doi.org/10.34293/economics.v7i4.618
Vishnu, C., Sajinkumar, K., Oommen, T., Coffman, R. A., Thrivikramji, K. P., Rani, V. R., … & Keerthy,
S. (2019). ‘Satellite-based assessment of the august 2018 flood in parts of Kerala, India’.
Geomatics, Natural Hazards and Risk, 10(1), pp. 758-767.
https://doi.org/10.1080/19475705.2018.1543212
Wang, S., Wang, H., Wu, Z., & Zhou, Y. (2021). ‘Using multi-factor analysis to predict urban flood
depth based on naive bayes’. Water, 13(4), p. 432. https://doi.org/10.3390/w13040432
Ward, P. J., Eisner, S., Flörke, M., Dettinger, M. D., & Kummu, M. (2014). ‘Annual flood sensitivities
to el niño–southern oscillation at the global scale’. Hydrology and Earth System Sciences, 18(1),
pp. 47-66. https://doi.org/10.5194/hess-18-47-2014
Wei, Z., Wang, K., Qu, H., Zhang, H., Bradfield, J. P., Kim, C., … & Hákonarson, H. (2009). ‘From
disease association to risk assessment: an optimistic view from genome-wide association studies
on type 1 diabetes’. PLoS Genetics, 5(10), p. e1000678.
https://doi.org/10.1371/journal.pgen.1000678
Wu, Z., Zhou, Y., & Wang, S. (2020). ‘Real-time prediction of the water accumulation process of
urban stormy accumulation points based on deep learning’. IEEE Access, 8, pp. 151938-151951.
https://doi.org/10.1109/access.2020.3017277
56

GEOG71200 Dissertation (MSc Geographical Information Science) Student ID: 11161619
Xie, X. and Shen, J. (2021). ‘Waterlogging resistance evaluation index and photosynthesis
characteristics selection: using machine learning methods to judge poplar’s waterlogging
resistance’. Mathematics, 9(13), p. 1542. https://doi.org/10.3390/math9131542
Yang, S. and Chang, L. (2020). ‘Regional inundation forecasting using machine learning techniques
with the internet of things’. Water, 12(6), p. 1578. https://doi.org/10.3390/w12061578
Yaseen, A., Lu, J., & Chen, X. (2022). ‘Flood susceptibility mapping in an arid region of Pakistan
through ensemble machine learning model’. Stochastic Environmental Research and Risk
Assessment, 36(10), pp. 3041-3061. https://doi.org/10.1007/s00477-022-02179-1
Yin, J., Yu, D., Yin, Z., Lü, M., & He, Q. (2016). ‘Evaluating the impact and risk of pluvial flash flood
on intra-urban road network: a case study in the city center of Shanghai, China’. Journal of
Hydrology, 537, pp. 138-145. https://doi.org/10.1016/j.jhydrol.2016.03.037
Yoon, D., Kim, Y. J., & Park, T. (2012). ‘Phenotype prediction from genome-wide association
studies: application to smoking behaviors’. BMC Systems Biology, 6(S2).
https://doi.org/10.1186/1752-0509-6-s2-s11
Yu, P. S., Chen, S. T., & Chang, I. (2006). ‘Support vector regression for real-time flood stage
forecasting’. Journal of Hydrology, 328(3-4), pp. 704-716.
https://doi.org/10.1016/j.jhydrol.2006.01.021
Zandbergen, P. A. (2008). ‘Applications of shuttle radar topography mission elevation data’.
Geography Compass, 2(5), pp. 1404-1431. https://doi.org/10.1111/j.1749-8198.2008.00154.x
Zhang, D., Peng, Q., Wang, D., Yang, T., Sorooshian, S., Liu, X., … & Zhuang, J. (2018). ‘Modeling
and simulating of reservoir operation using the artificial neural network, support vector
regression, deep learning algorithm’. Journal of Hydrology, 565, pp. 720-736.
https://doi.org/10.1016/j.jhydrol.2018.08.050
57
