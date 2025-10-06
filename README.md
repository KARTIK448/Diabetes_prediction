# RetinoScan Application

A Flask-based application for diabetic retinopathy detection using deep learning models.

## Prerequisites

- Docker and Docker Compose installed
- Git (optional, for cloning the repository)

## Quick Start

1. Clone or download this repository
2. Navigate to the project directory
3. Run the application using Docker Compose:

```bash
docker-compose up --build
```

4. Access the application at http://localhost:5000

## Docker Setup

This application is containerized using Docker with two services:

1. **app**: Flask application running on port 5000
2. **db**: PostgreSQL database running on port 5432

### Environment Variables

The application uses the following environment variables:

- `DATABASE_URL`: PostgreSQL connection string (set in docker-compose.yml)
- `FLASK_ENV`: Environment mode (production/development)

### Data Persistence

- PostgreSQL data is persisted in a Docker volume named `postgres_data`
- Uploaded files are persisted in a Docker volume named `uploads`

## Application Features

- User authentication (patient/doctor roles)
- Image upload and analysis using deep learning models
- Chat functionality between patients and doctors
- Patient profile management

## Models Used

The application uses three deep learning models for diabetic retinopathy detection:

1. ViT (Vision Transformer) Model
2. CNN (Convolutional Neural Network) Model
3. ResNet Model

## Development

To run in development mode:

```bash
docker-compose up --build
```

To stop the application:

```bash
docker-compose down
```

To stop and remove volumes (WARNING: This will delete all data):

```bash
docker-compose down -v
```

## Production Deployment

For production deployment, ensure:

1. Use a proper secret key for Flask
2. Use a proper database password
3. Consider using a reverse proxy like Nginx
4. Set up SSL certificates
5. Configure proper backup strategies for the database

## Troubleshooting

If you encounter issues:

1. Ensure Docker and Docker Compose are properly installed
2. Check that ports 5000 and 5432 are not being used by other applications
3. Check the logs with `docker-compose logs`
4. Ensure all required dependencies are installed (handled by Docker)
