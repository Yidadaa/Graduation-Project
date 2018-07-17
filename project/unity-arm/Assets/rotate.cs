using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Server;
using MyUtils;
using System.Threading;
using System;

public class rotate : MonoBehaviour {

  // Use this for initialization
  public Transform j_1;
  public Transform j_2;
  public Transform j_3;
  public GameObject point;
  public GameObject target;

  List<Vector3> initState;

  float[] rotateAngle = { 0f, 0f, 0f };

  float[] targetLimit = { 7f, 0f, 7f };

  HTTPServer s = new HTTPServer();
  Converter Converter = new Converter();

  void Start () {
    reset();
    // Thread t = new Thread(new ThreadStart(remoteControl));
    // t.Start();
    // initState.AddRange(new Vector3[]{ j_1.rotation.eulerAngles, j_2.rotation.eulerAngles, j_3.rotation.eulerAngles });
  }

  void Awake()
  {
    Application.targetFrameRate = 1000;
  }
  
  // Update is called once per frame
  void Update () {
    customControl();
    // print(getState());
    // remoteControl();
  }

  void customControl() {
    // 定义键盘的控制方式
    bool[] control = { Input.GetKey(KeyCode.Q), Input.GetKey(KeyCode.W), Input.GetKey(KeyCode.E) };
    bool[] direction = { Input.GetKey(KeyCode.LeftArrow), Input.GetKey(KeyCode.RightArrow) };
    float angle = (!direction[0] && !direction[1] ? 0 : direction[0] ? -1 : direction[1] ? 1 : 0) * 0.5f;
    float[] action = { 0f, 0f, 0f };

    bool shouldRandom = Input.GetKey (KeyCode.Space); // 按下空格键，机械臂随机运动
    bool shouldReset = Input.GetKeyDown (KeyCode.R); // 按下R键重置环境

    bool shouldAct = false;

    for (int i = 0; i < 3; i++) {
      action [i] = control [i] ? angle : 0f;
      shouldAct = shouldAct || control[i];
    }

    if (shouldRandom) {
      randomAction ();
    }

    if (shouldReset) {
      reset ();
    }

    if (shouldAct) {
      float[] state = step(action);
      print(Converter.float2string(state));
    }
  }

  void remoteControl() {
    s.run(step, reset);
  }

  float[] reset() {
    float[] newTargetPosition = { 0, 0, 0 };
    float a = random(targetLimit[0], -5f);
    float b = random(6.28f, 0);
    newTargetPosition[0] = a * (float)Math.Cos(b);
    newTargetPosition[2] = a * (float)Math.Sin(b);
    target.transform.position = new Vector3 (newTargetPosition[0], newTargetPosition[1] + 0.5f, newTargetPosition[2]);

    // j_1.rotation.SetEulerAngles(initState[0]);
    // j_2.rotation.SetEulerAngles(initState[1]);
    // j_3.rotation.SetEulerAngles(initState[2]);
    
    return getState();
  }

  float random(float scale, float offset) {
    return UnityEngine.Random.value * scale - offset;
  }

  float[] step(float[] action) {
    j_1.Rotate (0, action[0], 0);
    j_2.Rotate (0, 0, action[1]);
    j_3.Rotate (0, 0, action[2]);

    for (int i = 0; i < 3; i++) {
      rotateAngle[i] += action[i];
      rotateAngle[i] = rotateAngle[i] > 360 ? rotateAngle[i] - 360 :
        rotateAngle[i] < 0 ? -rotateAngle[i] : rotateAngle[i];
    }

    return getState();
  }

  void randomAction() {
    float scale = 5f;
    float offset = 2.5f;
    float[] angle = { random(scale, offset), random(scale, offset), random(scale, offset) };
    step (angle);
  }

  float getDistance() {
    return Vector3.Distance(point.transform.position, target.transform.position);
  }

  float[] getState() {
    float distance = getDistance();
    int done = distance < 1 ? 1 : 0; // 机械臂末端与目标点重合，任务完成
    List<float> state = new List<float>();
    state.Add(done);
    state.Add(distance);
    
    Transform[] ts = { j_1, j_2, j_3 };

    state.Add(ts[0].transform.rotation.eulerAngles[1] / 360f);
    state.Add(ts[1].transform.rotation.eulerAngles[2] / 360f);
    state.Add(ts[2].transform.rotation.eulerAngles[2] / 360f);

    //state.AddRange(new float[]{ rotateAngle[0] / 360, rotateAngle[1] / 360, rotateAngle[2] / 360 });

    var position = point.transform.position;
    var tp = target.transform.position;
    state.AddRange(new float[]{ position.x, position.y, position.z });
    state.AddRange(new float[]{ tp.x, tp.y, tp.z });

    return state.ToArray();
  }
}
