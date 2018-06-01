using System.Collections.Generic;

namespace MyUtils
{
  public class Converter
  {
    public float[] string2float(string s) {
			string[] res = s.Split (',');
			List<float> f = new List<float>();

			for(int i = 0; i < res.Length; i++){
				f.Add(System.Convert.ToSingle(res[i]));
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