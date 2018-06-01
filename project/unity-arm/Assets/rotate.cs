using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Server;
using MyUtils;
using System.Threading;

public class rotate : MonoBehaviour {

  // Use this for initialization
  public Transform j_1;
  public Transform j_2;
  public Transform j_3;
  public GameObject point;
  public GameObject target;

  float[] targetLimit = { 20f, 0f, 20f };

  HTTPServer s = new HTTPServer();
  Converter Converter = new Converter();

  void Start () {
    reset();
    // Thread t = new Thread(new ThreadStart(remoteControl));
    // t.Start();
  }
  
  // Update is called once per frame
  void Update () {
    customControl();
    remoteControl();
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
    for (int i = 0; i < 3; i++) {
      newTargetPosition [i] = random(targetLimit [i], targetLimit[i] / 2);
    }
    target.transform.position = new Vector3 (newTargetPosition[0], newTargetPosition[1] + 0.5f, newTargetPosition[2]);
    
    return getState();
  }

  float random(float scale, float offset) {
    return Random.value * scale - offset;
  }

  float[] step(float[] action) {
    j_1.Rotate (0, action[0], 0);
    j_2.Rotate (0, 0, action[1]);
    j_3.Rotate (0, 0, action[2]);

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
    int done = distance < 0.5 ? 1 : 0; // 机械臂末端与目标点重合，任务完成
    List<float> state = new List<float>();
    state.Add(done);
    state.Add(distance);
    
    Transform[] ts = { j_1, j_2, j_3 };

    for (int i = 0; i < 3; i++) {
      Quaternion q = ts[i].rotation;
      state.AddRange(new float[]{ q.x, q.y, q.z, q.w });
    }

    var position = point.transform.position;
    state.AddRange(new float[]{ position.x, position.y, position.z });

    return state.ToArray();
  }
}
