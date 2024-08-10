import torch.nn.functional as F


def knowledge_distillation(student_model, teacher_model, data_loader, optimizer, temperature=5.0, alpha=0.7,
                           num_epochs=5):
    """
    Performs knowledge distillation to train a student model using a teacher model.

    Args:
        student_model (nn.Module): The smaller, student model to be trained.
        teacher_model (nn.Module): The larger, pre-trained teacher model.
        data_loader (DataLoader): DataLoader for the training dataset.
        optimizer (Optimizer): Optimizer for the student model.
        temperature (float): Temperature to soften the logits from the teacher model.
        alpha (float): Weighting factor for the distillation loss vs. original loss.
        num_epochs (int): Number of epochs to train the student model.

    Returns:
        None
    """
    teacher_model.eval()  # Set the teacher model to evaluation mode

    for epoch in range(num_epochs):
        student_model.train()  # Set the student model to training mode
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()

            # Forward pass through both teacher and student models
            teacher_logits = teacher_model(data).detach()  # Teacher's output (no gradient)
            student_logits = student_model(data)  # Student's output

            # Compute the distillation loss
            distill_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature * temperature)

            # Compute the standard cross-entropy loss
            student_loss = F.cross_entropy(student_logits, target)

            # Combine losses
            loss = alpha * distill_loss + (1. - alpha) * student_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print loss for each epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Example usage (assuming you have defined your student and teacher models, data_loader, and optimizer):
# knowledge_distillation(student_model, teacher_model, train_loader, optimizer)
