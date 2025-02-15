let studObj={name:"Jane Doe",year:"2nd"};
console.log(studObj);
//JSON-java script object notation
//used when sending/receviing data from server
let studob=JSON.stringify(studObj);
console.log(studob);


{
    "coord": {
      "lon": -0.1257,
      "lat": 51.5085
    },
    "weather": [
      {
        "id": 800,
        "main": "Clear",
        "description": "clear sky",
        "icon": "01d"
      }
    ],
    "base": "stations",
    "main": {
      "temp": 12.39,
      "feels_like": 11.93,
      "temp_min": 10.86,
      "temp_max": 13.47,
      "pressure": 1028,
      "humidity": 86,
      "sea_level": 1028,
      "grnd_level": 1025
    },
    "visibility": 10000,
    "wind": {
      "speed": 3.6,
      "deg": 240
    },
    "clouds": {
      "all": 7
    },
    "dt": 1729590538,
    "sys": {
      "type": 2,
      "id": 2075535,
      "country": "GB",
      "sunrise": 1729579011,
      "sunset": 1729615961
    },
    "timezone": 3600,
    "id": 2643743,
    "name": "London",
    "cod": 200
  }
  {
    "coord": {
      "lon": 139.6917,
      "lat": 35.6895
    },
    "weather": [
      {
        "id": 803,
        "main": "Clouds",
        "description": "broken clouds",
        "icon": "04n"
      }
    ],
    "base": "stations",
    "main": {
      "temp": 21.79,
      "feels_like": 22.04,
      "temp_min": 20.34,
      "temp_max": 23.1,
      "pressure": 1022,
      "humidity": 77,
      "sea_level": 1022,
      "grnd_level": 1017
    },
    "visibility": 10000,
    "wind": {
      "speed": 2.06,
      "deg": 30
    },
    "clouds": {
      "all": 75
    },
    "dt": 1729590174,
    "sys": {
      "type": 2,
      "id": 268105,
      "country": "JP",
      "sunrise": 1729544019,
      "sunset": 1729583848
    },
    "timezone": 32400,
    "id": 1850144,
    "name": "Tokyo",
    "cod": 200
  }
  try{
    let response=fetch(url)
    let data=response.json();
    displayData(data);
    console.log(data);
  }
  catch(error){
    console.log("an error occured while fetching data please check city name again")
  }