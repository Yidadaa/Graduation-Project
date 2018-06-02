using System.Collections.Generic;
using System;
using UnityEngine;

namespace MyUtils
{
  public class Converter
  {
    public float[] string2float(string s) {
			string[] res = s.Split (',');
			List<float> f = new List<float>();

			for(int i = 0; i < res.Length; i++){
        try
        {
            f.Add(System.Convert.ToSingle(res[i]));
        }
        catch (System.Exception)
        {
            Debug.Log(res[i]);
            throw;
        }
			}

      return f.ToArray();
    }

    public string float2string(float[] f) {
      List<string> s = new List<string>();

      foreach (var item in f) {
          s.Add(item.ToString());
      }

      return string.Join(",", s.ToArray());
    }
  }
}