using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Text;

using UnityEngine;
using MyUtils;

namespace Server
{
	public class HTTPServer
	{
		public delegate float[] CallbackTypeOfStep(float[] action);
		public delegate float[] CallbackTypeOfReset();
		HttpListener listener = new HttpListener ();
		Converter Converter = new Converter();
		Boolean shutDown = false;

		private string port = "88";

		public HTTPServer (string port)
		{
			listener.AuthenticationSchemes = AuthenticationSchemes.Anonymous;
			listener.Prefixes.Add ("http://127.0.0.1:" + port + "/");
		}

		public HTTPServer ()
		{
			listener.AuthenticationSchemes = AuthenticationSchemes.Anonymous;
			listener.Prefixes.Add ("http://127.0.0.1:" + port + "/");
		}

		public void run(CallbackTypeOfStep stepFunc, CallbackTypeOfReset resetFunc) {
			if (shutDown)
				return;

			listener.Start ();
			Debug.Log ("Listening: port " + port);

			HttpListenerContext ctx = listener.GetContext ();
			Stream stream = ctx.Request.InputStream;
			System.IO.StreamReader reader = new System.IO.StreamReader (stream, ASCIIEncoding.UTF8);
			String body = reader.ReadToEnd ();

			string respStr = "";

			if (body.Equals ("exit")) {
				shutDown = true;
			} else if (body.Equals ("reset")) {
				var state = resetFunc();
				respStr = Converter.float2string(state);
			} else {
				var action = Converter.string2float(body);
				var state = stepFunc(action);
				respStr = Converter.float2string(state);
			}

			Debug.Log ("Recieved: " + body);

			HttpListenerResponse resp = ctx.Response;

			byte[] buffer = ASCIIEncoding.UTF8.GetBytes (respStr);
			resp.ContentLength64 = buffer.Length;

			Stream output = resp.OutputStream;
			output.Write (buffer, 0, buffer.Length);
			output.Close ();
		}
	}
}