#include <AccelStepper.h>
#include <MultiStepper.h>

// --- Pin Definitions ---
#define DIR1 8
#define PUL1 9
#define DIR2 10
#define PUL2 11
#define DIR3 12
#define PUL3 13
#define EN3 7
#define RELAY_PIN 2

// --- Constants (Steps per Degree) ---
const float SPD1 = (3200.0 * 6.0) / 360.0;
const float SPD2 = (1600.0 * 3.0) / 360.0;
const float SPD3 = (1600.0 * 3.0) / 360.0;

// --- Mechanical Angle Limits ---
const float MIN_ANG1 = -180.0, MAX_ANG1 = 180.0;
const float MIN_ANG2 = -25.0,  MAX_ANG2 = 110.0;
const float MIN_ANG3 = -135.0, MAX_ANG3 = 20.0;

// --- Absolute Position Tracking ---
// The Robot should start at these angles
float current_angle1 = 0.0;
float current_angle2 = 110.0;
float current_angle3 = 20.0;

// --- Initialize Objects ---
AccelStepper stepper1(AccelStepper::DRIVER, PUL1, DIR1);
AccelStepper stepper2(AccelStepper::DRIVER, PUL2, DIR2);
AccelStepper stepper3(AccelStepper::DRIVER, PUL3, DIR3);
MultiStepper syncSteppers;

void printCurrentAngles() {
  Serial.print("Angles -> Base: "); Serial.print(current_angle1);
  Serial.print(" | Shoulder: "); Serial.print(current_angle2);
  Serial.print(" | Elbow: "); Serial.println(current_angle3);
}

void setup() {
  Serial.begin(9600);

  stepper1.setPinsInverted(false, false, false);
  stepper2.setPinsInverted(true, false, false);
  stepper3.setPinsInverted(true, false, false);

  stepper1.setMaxSpeed(1500);
  stepper1.setAcceleration(500);
  stepper2.setMaxSpeed(600);
  stepper2.setAcceleration(300.0);
  stepper3.setMaxSpeed(400);
  stepper3.setAcceleration(300);

  stepper1.setCurrentPosition(round(current_angle1 * SPD1));
  stepper2.setCurrentPosition(round(current_angle2 * SPD2));
  stepper3.setCurrentPosition(round(current_angle3 * SPD3));

  syncSteppers.addStepper(stepper2);
  syncSteppers.addStepper(stepper3);

  pinMode(EN3, OUTPUT);
  digitalWrite(EN3, LOW);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);

  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    // Ignore stray newline characters
    if (cmd == '\n' || cmd == '\r' || cmd == ' ') return; 

    // Parse the float directly from the serial stream (minimizes RAM/parsing)
    float target = 0.0;
    if (cmd == '1' || cmd == '2' || cmd == '3' || cmd == 'R' || cmd == 'r') {
      target = Serial.parseFloat();
    }
    
    // Clear any remaining characters up to the newline
    while(Serial.available() && Serial.read() != '\n');

    switch (cmd) {
      case '1': {
        if (target >= MIN_ANG1 && target <= MAX_ANG1) {
          stepper1.move(round((target - current_angle1) * SPD1));
          stepper1.runToPosition();
          current_angle1 = target;
          printCurrentAngles();
        } else {
          Serial.println("ERR: Limits");
        }
        Serial.println("DONE");
        break;
      }

      case '2': {
        if (target >= MIN_ANG2 && target <= MAX_ANG2) {
          float delta2 = target - current_angle2;
          long positions[2];
          positions[0] = stepper2.currentPosition() + round(delta2 * SPD2);
          positions[1] = stepper3.currentPosition() + round((-delta2 / 3.0) * SPD3);
          
          syncSteppers.moveTo(positions);
          syncSteppers.runSpeedToPosition();
          
          current_angle2 = target;
          printCurrentAngles();
        } else {
          Serial.println("ERR: Limits");
        }
        Serial.println("DONE");
        break;
      }

      case '3': {
        if (target >= MIN_ANG3 && target <= MAX_ANG3) {
          stepper3.move(round((target - current_angle3) * SPD3));
          stepper3.runToPosition();
          current_angle3 = target;
          printCurrentAngles();
        } else {
          Serial.println("ERR: Limits");
        }
        Serial.println("DONE");
        break;
      }

      case 'H':
      case 'h': {
        // Home Base
        stepper1.move(round((0.0 - current_angle1) * SPD1));
        stepper1.runToPosition();
        current_angle1 = 0.0;

        // Home Shoulder (Sync logic)
        float delta2 = 60.0 - current_angle2;
        long positions[2];
        positions[0] = stepper2.currentPosition() + round(delta2 * SPD2);
        positions[1] = stepper3.currentPosition() + round((-delta2 / 3.0) * SPD3);
        syncSteppers.moveTo(positions);
        syncSteppers.runSpeedToPosition();
        current_angle2 = 60.0;

        // Home Elbow
        stepper3.move(round((-135.0 - current_angle3) * SPD3));
        stepper3.runToPosition();
        current_angle3 = -135.0;

        printCurrentAngles();
        Serial.println("DONE");
        break;
      }

      case 'R':
      case 'r':
        digitalWrite(RELAY_PIN, target > 0 ? HIGH : LOW);
        Serial.println("DONE");
        break;
    }
  }
}