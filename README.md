# Smart Self-Balancing Robot

A self-balancing robot based on the inverted pendulum model, combining real-time sensor feedback, computer vision, and voice-assisted interaction.

## Overview

This project uses sensor data and control algorithms to maintain balance on two wheels. A Raspberry Pi handles computer vision and intelligent features, while an Arduino manages real-time control of the balancing system.

## Features

* Self-balancing using sensor feedback
* Inverted pendulum control system
* Real-time motor control
* Computer vision using OpenCV
* Voice-enabled smart assistant
* Integration between Arduino and Raspberry Pi

## Hardware

* Arduino Uno
* Raspberry Pi 4
* MPU6050 IMU
* Nema-17 Stepper Motors
* DRV8825 Motor Drivers
* 3300mAh LiPo Battery

## Software

* Python
* Embedded C
* OpenCV
* Gemini API

## System Architecture

graph TD
    A[Raspberry Pi] --> |1. High-Level Decisions| B(Arduino Microcontroller);
    subgraph Sensor Input
        C[MPU6050 IMU] --> A;
        D[Camera Feed] --> A;
    end

    A -- Processes: --> CV(Computer Vision) & VA(Voice Assistant);
    B -- Controls Signals to --> E{Motor Drivers};
    E --> F[Stepper Motors];

    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333


## Key Concepts

* Inverted Pendulum Model
* Feedback Control Systems
* Sensor Fusion
* Computer Vision
* Human-Robot Interaction

## Future Improvements

* Autonomous navigation
* SLAM integration
* Improved obstacle avoidance
* Remote monitoring interface

