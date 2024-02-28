# Tag the image so you can push it to AWS ECR (private repo)
docker tag mltest:latest "$ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/mltest:latest

# Push the image
docker push "$ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/mltest:latest


<script>  

UPLOAD IMAGE
    document.getElementById('upload-form').onsubmit = function(e){  
        e.preventDefault();  
        var formData = new FormData();  
        var imageFile = document.getElementById('image-input').files[0];  
        formData.append("file", imageFile);  
        fetch('/invocations', {  
            method: 'POST',  
            body: formData,  
        })  
        .then(response => response.json())  
        .then(data => console.log(data))  
        .catch(error => console.error('Error:', error));  
    };  
</script>  