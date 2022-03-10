<!-- PROJECT LOGO -->
<br />
<p align="center">
   <img src="https://github.com/martindoff/DC-TMPC/tree/master/plot/tmpc-traj.jpg" alt="Logo" width="300" height="300">
  <p align="center">
   DC-TMPC: A tube-based MPC algorithm for systems that can be expressed as a difference of convex functions. 
    <br />  
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Developed at the University of Oxford, DC-TMPC is a novel robust tube-based nonlinear model 
predictive control paradigm for nonlinear systems whose dynamics can be expressed as a difference of convex functions. 
The approach relies on successively perturbing the system predicted trajectories and bounding
the linearisation error by exploiting convexity of the system dynamics. The linearisation error is then treated as a
disturbance of the perturbed system to construct robust tubes containing the predicted trajectories, enabling the
robust nonlinear MPC optimisation to be performed in real time as a sequence of convex optimisation programs. 

The present implementation involves regulating a coupled tank whose dynamics can be represented as a difference of convex functions. 

### Built With

* Python 3
* CVX
* Mosek



<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

You need to install the following:
* numpy
* scipy
* matplotlib
* [cvxpy](https://www.cvxpy.org/install/index.html)
* [mosek](https://www.mosek.com/downloads/)

Run the following command to install all modules at once

   ```sh
   pip3 install numpy scipy matplotlib cvxpy mosek
   ```

In order to use mosek, you will need a license. Look [here] (https://www.mosek.com/products/academic-licenses/) to set it up. 

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/martindoff/DC-TMPC.git
   ```
2. Go to directory 
   ```sh
   cd DC-TMPC-master
   ```
3. Run the program
   ```sh
   python3 main.py
   ```

<!-- ROADMAP -->
## Roadmap


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- CONTACT -->
## Contact

Martin Doff-Sotta - martin.doff-sotta@eng.ox.ac.uk

Linkedin: https://www.linkedin.com/in/mdoffsotta/



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
