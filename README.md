# Machine-Learning-Based-Emergency-Mode-Traffic-Signal-Simulation
This project successfully demonstrates how deep learning can be applied to real-world traffic management problems. By integrating ambulance detection with automatic traffic signal control, the system provides an efficient and scalable solution for emergency vehicle prioritization. With further enhancements such as live CCTV integration and hardware deployment, this system can be implemented in real-world smart traffic environments.

## About

This project presents an intelligent traffic management system that automatically detects ambulances using deep learning and dynamically controls traffic signals to provide priority passage. The system aims to reduce emergency response time by ensuring ambulances receive immediate green signals at traffic intersections.

The project uses a YOLO-based object detection model trained specifically to identify ambulances from images or video feeds. Once an ambulance is detected, the system overrides the normal traffic signal cycle and switches the signal to green for a fixed duration, allowing the emergency vehicle to pass safely and without delay.

## Features

• Automated Ambulance Detection
The system uses a deep learning–based YOLO model to automatically detect ambulances from input images or video frames without human intervention.

• Real-Time Traffic Signal Control
Traffic signals are controlled dynamically based on the detection results, enabling immediate response when an ambulance is identified.

• Priority Signal Override
Upon detecting an ambulance, the system overrides the normal traffic signal cycle and switches the signal to green to allow emergency vehicles to pass smoothly.

• Normal Traffic Cycle Management
In the absence of ambulances, the system follows a standard traffic signal sequence (Red → Yellow → Green) ensuring regular traffic flow.

• Web-Based Graphical User Interface
A Streamlit-based web application displays detection results along with real traffic signal images for better visualization and user interaction.

• Image-Based Input Support
Users can upload images for testing and validation, making the system easy to demonstrate and evaluate.

• Visual Traffic Signal Representation
The traffic signal status is shown using realistic traffic light images, improving clarity and understanding of system behavior.

• Automatic Recovery to Normal Mode
After ambulance passage, the system automatically switches back to the normal traffic signal cycle.

• High Detection Accuracy
The trained YOLO model provides fast and accurate ambulance detection even in complex traffic scenes.

• Scalable and Extendable Design
The system can be easily extended to support video streams, live CCTV feeds, or hardware-based traffic controllers.

## Requirements

• Operating System
Windows / Linux / macOS

• Programming Language
Python 3.8 or above

• Development Environment
Visual Studio Code (VS Code)

• Deep Learning Framework
PyTorch

• Object Detection Model
YOLO (You Only Look Once)

• Web Application Framework
Streamlit

• Image Processing Library
OpenCV

• Supporting Libraries
NumPy, Pillow

• Dataset Annotation Tool
LabelImg / Roboflow

• Processor
Intel i5 or higher (or equivalent)

• RAM
Minimum 8 GB (16 GB recommended)

• Storage
At least 10 GB free space

• GPU (Optional but Recommended)
NVIDIA GPU with CUDA support for faster training and inference

• Input Devices
Camera / CCTV footage / Image files

## System Architecture

<img width="1536" height="1024" alt="architecture" src="https://github.com/user-attachments/assets/177640ee-6080-48a9-baf1-04ae9772b0dc" />


<img width="1536" height="1024" alt="data flow diagram" src="https://github.com/user-attachments/assets/5811d1c1-bdc6-49a3-a457-5f3aebe46cfc" />


## Output 

<img width="1919" height="909" alt="image" src="https://github.com/user-attachments/assets/b43930ca-e6ff-4691-a0bf-ac42754ccf13" />


<img width="1918" height="906" alt="image" src="https://github.com/user-attachments/assets/01cca917-e190-48ec-882d-7404a0685c9c" />


<img width="1919" height="906" alt="image" src="https://github.com/user-attachments/assets/2daabf6a-4d79-445d-a385-d5828a87f6e9" />

## Results and Impact

The implemented system successfully detects ambulances from input images using a trained YOLO-based deep learning model. The model demonstrates high detection accuracy across various traffic scenes, including different lighting conditions and background vehicle densities. During testing, the system correctly identified ambulances and triggered an immediate traffic signal override. Upon detection, the signal switched to green within milliseconds, allowing the emergency vehicle to pass without interruption. After the override duration, the signal reliably returned to the normal traffic cycle without manual intervention.

This project has a significant impact on emergency response efficiency by reducing delays caused by traffic congestion. Faster ambulance movement directly contributes to improved patient survival rates and timely medical care. The system promotes smart traffic management by integrating artificial intelligence into traditional traffic control mechanisms. It reduces the dependency on human operators and minimizes errors caused by manual signal handling.

## References

[1] H. Shi and C. Liu, “A new foreground segmentation method for video analysis in different color
spaces,” in 24th International Conference on Pattern Recognition, IEEE, 2018.

[2] G. Liu, H. Shi, A. Kiani, A. Khreishah, J. Lee, N. Ansari, C. Liu, and M. M. Yousef, “Smart traffic
monitoring system using computer vision and edge computing,” IEEE Transactions on Intelligent
Transportation Systems, 2021.

[3] H. Ghahremannezhad, H. Shi, and C. Liu, “Automatic road detection in traffic videos,” in 2020 IEEE
Intl Conf on Parallel & Distributed Processing with Applications, Big Data & Cloud Computing, Sustainable
Computing & Communications, Social Computing & Networking(ISPA/BDCloud/SocialCom/ SustainCom),
pp. 777–784, IEEE, 2020.

[4] H. Ghahremannezhad, H. Shi, and C. Liu, “A new adaptive bidirectional region-of-interest detection
method for intelligent traffic video analysis,” in 2020 IEEE Third International Conference on Artificial
Intelligence and Knowledge Engineering (AIKE), pp. 17–24, IEEE, 2020.

[5] H. Ghahremannezhad, H. Shi, and C. Liu, “Robust road region extraction in video under various
illumination and weather conditions,” in 2020 IEEE 4th International Conference on Image Processing,
Applications and Systems (IPAS), pp. 186–191, IEEE, 2020.
[6] H. Shi, H. Ghahremannezhadand, and C. Liu, “A statistical modeling method for road recognition
in traffic video analytics,” in 2020 11th IEEE International Conference on Cognitive Infocommunications
(CogInfoCom), pp. 000097–000102, IEEE, 2020.

[7] H. Ghahremannezhad, H. Shi, and C. Liu, “A real time accident detection framework for traffic
video analysis,” in Machine Learning and Data Mining in Pattern Recognition, MLDM, pp. 77–92, ibai
publishing, Leipzig, 2020.

[8] M. O. Faruque, H. Ghahremannezhad, and C. Liu, “Vehicle classification in video using deep
learning,” in Machine Learning and Data Mining in Pattern Recognition, MLDM, pp. 117–131, ibai
publishing, Leipzig, 2019.

[9] H. Ghahremannezhad, H. Shi, and C. Liu, “A new online approach for moving cast shadow
suppression in traffic videos,” in 2021 IEEE International Intelligent Transportation Systems Conference
(ITSC), pp. 3034–3039, IEEE, 2021.

[10] H. Shi, H. Ghahremannezhad, and C. Liu, “Anomalous driving detection for traffic surveillance video
analysis,” in 2021 IEEE International Conference on Imaging Systems and Techniques (IST), pp. 1–6,
IEEE, 2021.

[11] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “Yolov4: Optimal speed and accuracy of object
detection,” arXiv preprint arXiv:2004.10934, 2020.

[12] H. W. Kuhn, “The hungarian method for the assignment problem,” Naval research logistics
quarterly, vol. 2, no. 1-2, pp. 83–97, 1955.

[13] .L. Zheng, Z. Peng, J. Yan, and W. Han, ‘‘An online learning and unsupervised traffic anomaly
detection system,’’ Adv. Sci. Lett., vol. 7, no. 1, pp. 449–455, 2012.

[14] Y. Fangchun, W. Shangguang, L. Jinglin, L. Zhihan, and S. Qibo, ‘‘An overview of Internet of
vehicles,’’ China Commun., vol. 11, no. 10, pp. 1–15, Oct. 2014.

[15] .C. Ma, W. Hao, A. Wang, and H. Zhao, ‘‘Developing a coordinated signal control system for urban
ring road under the vehicle-infrastructure connected environment,’’ IEEE Access, vol. 6, pp. 52471–52478,
2018.

[16] S. Zhang, J. Chen, F. Lyu, N. Cheng, W. Shi, and X. Shen, ‘‘Vehicular communication networks in
the automated driving era,’’ IEEE Commun. Mag., vol. 56, no. 9, pp. 26–32, Sep. 2018.

[17] Y. Wang, D. Zhang, Y. Liu, B. Dai, and L. H. Lee, ‘‘Enhancing transportation systems via deep
learning: A survey,’’ Transp. Res. C, Emerg. Technol., 2018.

[18] G. Wu, F. Chen, X. Pan, M. Xu, and X. Zhu, ‘‘Using the visual intervention influence of pavement
markings for rutting mitigation—Part I: Preliminary experiments and field tests,’’ Int. J. Pavement Eng., vol.
20, no. 6, pp. 734– 746, 2019.

[19] S. Ramos, S. Gehrig, P. Pinggera, U. Franke, and C. Rother, ‘‘Detecting unexpected obstacles for
self-driving cars: Fusing deep learning and geometric modeling,’’ in Proc. IEEE Intell. Vehicles Symp. (IV),
Jun. 2017, pp. 1025– 1032.

[20] T. Qu, Q. Zhang, and S. Sun, ‘‘Vehicle detection from high-resolution aerial images using spatial
pyramid pooling-based deep convolutional neural networks,’’ Multimedia Tools Appl., vol. 76, no. 20, pp.
21651–21663, 2017. object tracking: A literature review,” Artificial Intelligence, vol. 293, p. 103448, 2021.

[21] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, “Simple online and realtime tracking,” in 2016
IEEE international conference on image processing (ICIP), pp. 3464–3468, IEEE, 2016.

[22] “Nvidia ai city challenge – data and evaluation.” https://www. aicitychallenge.org/2022-data-andevaluation/. Accessed: 2022-04-27.

[23] L. Yue, M. Abdel-Aty, Y. Wu, O. Zheng, and J. Yuan, “In-depth approach for identifying crash
causation patterns and its implications for pedestrian crash prevention,” Journal of safety research, vol. 73,
pp. 119–132, 2020.

[24] K. Dresner and P. Stone, “A Multiagent Approach to Autonomous Intersection Management,” Journal of
Artificial Intelligence Research, vol. 31, pp. 591–656, 2008.

[25] S. Pandit, D. Ghosal, M. Zhang, and C.-N. Chuah, “Adaptive Traffic Signal Control with Deep
Reinforcement Learning,” IEEE Transactions on Intelligent Transportation Systems, 2019.

[26] M. Wiering, J. van Veenen, J. Vreeken and A. Koopman, “Intelligent Traffic Light Control,” Institute of
Information and Computing Sciences, Utrecht University, 2004.

[27] Y. Wei et al., “IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control,”
Proceedings of the 24th ACM SIGKDD Conference, 2018.

[28] A. Sharma and S. Singh, “Machine Learning Based Traffic Signal Control for Emergency Vehicles,”
International Journal of Advanced Research in Computer Engineering & Technology, 2021.

[29] S. Mousavi, M. Schukat, and E. Howley, “Traffic Light Control Using Deep Policy-Gradient and ValueFunction Based Reinforcement Learning,” IET Intelligent Transport Systems, 2017.

