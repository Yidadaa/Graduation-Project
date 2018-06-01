using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Web;
using UnityEngine;

namespace AssemblyCSharp
{
	public class server
	{
		public delegate void Callback(float[] angle);
		public delegate float[] CallbackWithFloat();
		HttpListener listener = new HttpListener ();
		Boolean shutDown = false;

		public server ()
		{
			listener.AuthenticationSchemes = AuthenticationSchemes.Anonymous;
			listener.Prefixes.Add ("http://127.0.0.1:8888/");
		}

		public void run(Callback setFunc, CallbackWithFloat getFunc) {
			if (shutDown)
				return;

			listener.Start ();
			Debug.Log ("Server端已启动...");

			HttpListenerContext ctx = listener.GetContext ();
			Stream stream = ctx.Request.InputStream;
			System.IO.StreamReader reader = new System.IO.StreamReader (stream, ASCIIEncoding.UTF8);
			String body = reader.ReadToEnd ();

			if (body.Equals ("exit")) {
				shutDown = true;
			}

			string[] res = body.Split (' ');
			float[] action = new float[]{ 0, 0, 0, 0, 0, 0 };

			for(int i = 0; i < res.Length; i++){
				action[i] = Convert.ToSingle(res[i]);
			}

			Debug.Log (body);
			setFunc (action);

			HttpListenerResponse resp = ctx.Response;
			string resString = "";

			float[] angle = getFunc ();

			foreach (float i in angle) {
				resString += " " + i.ToString ();
			}

			byte[] buffer = ASCIIEncoding.UTF8.GetBytes (resString);
			resp.ContentLength64 = buffer.Length;
			Stream output = resp.OutputStream;
			output.Write (buffer, 0, buffer.Length);
			output.Close ();
		}
	}
}