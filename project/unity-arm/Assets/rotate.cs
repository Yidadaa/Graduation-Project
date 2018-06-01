using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class rotate : MonoBehaviour {

  // Use this for initialization
  public Transform j_1;
  public Transform j_2;
  public Transform j_3;
  public Rigidbody a_1;
  public Rigidbody a_2;
  public Rigidbody a_3;
  public GameObject point;
  public GameObject target;

  float[] targetLimit = { 12f, 12f, 12f };

  void Start () {
    
  }
  
  // Update is called once per frame
  void Update () {
    bool[] control = { Input.GetKey(KeyCode.Q), Input.GetKey(KeyCode.W), Input.GetKey(KeyCode.E) };
    bool[] direction = { Input.GetKey(KeyCode.LeftArrow), Input.GetKey(KeyCode.RightArrow) };
    float angle = (!direction[0] && !direction[1] ? 0 : direction[0] ? -1 : direction[1] ? 1 : 0) * 0.5f;
    float[] action = { 0f, 0f, 0f };

    bool shouldRandom = Input.GetKey (KeyCode.Space); // 按下空格键，机械臂随机运动
    bool shouldReset = Input.GetKeyDown (KeyCode.R); // 按下R键重置环境

    for (int i = 0; i < 3; i++) {
      action [i] = control [i] ? angle : 0f;
    }

    if (shouldRandom) {
      randomAction ();
    }

    if (shouldReset) {
      reset ();
    }
  
    step (action);

    print (Vector3.Distance(point.transform.position, target.transform.position));
  }

  void reset() {
    float[] newTargetPosition = { 0, 0, 0 };
    for (int i = 0; i < 3; i++) {
      newTargetPosition [i] = random(targetLimit [i], 0f);
    }
    target.transform.position = new Vector3 (newTargetPosition[0], newTargetPosition[1], newTargetPosition[2]);
  }

  float random(float scale, float offset) {
    return Random.value * scale - offset;
  }

  void step(float[] action) {
    j_1.Rotate (0, action[0], 0);
    j_2.Rotate (0, 0, action[1]);
    j_3.Rotate (0, 0, action[2]);
  }

  void randomAction() {
    float scale = 5f;
    float offset = 2.5f;
    float[] angle = { random(scale, offset), random(scale, offset), random(scale, offset) };
    step (angle);
  }
}
