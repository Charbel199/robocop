using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.Threading.Tasks;
using VehicleBehaviour;

public class FuzzyController : MonoBehaviour
{
    [SerializeField] Transform target;
    [SerializeField] WheelVehicle rover;
    public float FUZZY_MAX_SPEED = 50;
    private float distance, deviation, motor_r_speed, motor_l_speed, left_percentage, right_percentage, _throttle, _steering;
    private Vector3 self;
    private Vector3 other;

    void Start()
    {
        StartCoroutine(GetFuzzyControl("url"));
        rover.Throttle = 50;
        rover.Steering = 50;
    }

    IEnumerator GetFuzzyControl(string url)
    {
        while (true)
        {
            self = gameObject.transform.position;
            other = target.position;

            distance = other.z - self.z;
            deviation = Mathf.Atan2(other.x - self.x, distance);

            using (UnityWebRequest request = UnityWebRequest.Get(url + distance + deviation))
            {
                yield return request.SendWebRequest();
                if (request.result == UnityWebRequest.Result.Success)
                {
                    string response = request.downloadHandler.text;
                    Debug.Log(response);
                    ControlResponse controlResponse = ControlResponse.CreateFromJSON(response);
                    motor_r_speed = controlResponse.motor_right;
                    motor_l_speed = controlResponse.motor_left;
                    _throttle = (motor_l_speed + motor_r_speed) / 2;
                    _steering = motor_r_speed - motor_l_speed;
                    rover.Throttle = _throttle / FUZZY_MAX_SPEED;
                    rover.Steering = _steering / FUZZY_MAX_SPEED;
                }
                else{
                    rover.Throttle = 25/FUZZY_MAX_SPEED;
                    rover.Steering = 25/FUZZY_MAX_SPEED;
                }
            }
        }
    }
}

public class ControlResponse
{
    public float motor_right;
    public float motor_left;

    public static ControlResponse CreateFromJSON(string jsonString)
    {
        return JsonUtility.FromJson<ControlResponse>(jsonString);
    }
}
