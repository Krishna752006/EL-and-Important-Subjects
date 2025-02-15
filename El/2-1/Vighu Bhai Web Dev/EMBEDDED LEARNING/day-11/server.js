const express = require('express');
const app = express();
const cors = require('cors');
const port = 7654;

app.use(cors()); 
const weatherData = {
    london: {
        coord: {
            lon: -0.1257,
            lat: 51.5085
        },
        weather: [
            {
                id: 800,
                main: "Clear",
                description: "clear sky",
                icon: "01d"
            }
        ],
        base: "stations",
        main: {
            temp: 12.39,
            feels_like: 11.93,
            temp_min: 10.86,
            temp_max: 13.47,
            pressure: 1028,
            humidity: 86,
            sea_level: 1028,
            grnd_level: 1025
        },
        visibility: 10000,
        wind: {
            speed: 3.6,
            deg: 240
        },
        clouds: {
            all: 7
        },
        dt: 1729590538,
        sys: {
            type: 2,
            id: 2075535,
            country: "GB",
            sunrise: 1729579011,
            sunset: 1729615961
        },
        timezone: 3600,
        id: 2643743,
        name: "London",
        cod: 200
    },
    newyork: {
        coord: {
            lon: -74.006,
            lat: 40.7128
        },
        weather: [
            {
                id: 801,
                main: "Clouds",
                description: "few clouds",
                icon: "02d"
            }
        ],
        base: "stations",
        main: {
            temp: 20.0,
            feels_like: 19.5,
            temp_min: 18.5,
            temp_max: 22.0,
            pressure: 1015,
            humidity: 70,
            sea_level: 1015,
            grnd_level: 1012
        },
        visibility: 9000,
        wind: {
            speed: 5.0,x
            deg: 180
        },
        clouds: {
            all: 20
        },
        dt: 1729590538,
        sys: {
            type: 2,
            id: 2075535,
            country: "US",
            sunrise: 1729581011,
            sunset: 1729627011
        },
        timezone: -14400,
        id: 5128581,
        name: "New York",
        cod: 200
    }
};
app.get('/weather/:city', (req, res) => {
    const city = req.params.city.toLowerCase(); 
    const data = weatherData[city];

    if (data) {
        res.json(data); 
    } else {
        res.status(404).json({ error: 'City not found' })
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}/weather`);
});
