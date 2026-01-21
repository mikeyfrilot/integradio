"""
Integration tests for form validation components.

Tests focus on real validation behavior and integration between
validators and form components.
"""

import pytest
import re
from integradio.ux.validation import (
    ValidationState,
    ValidationResult,
    ValidationRule,
    FieldValidator,
    VALIDATION_CSS,
    validate_required,
    validate_email,
    validate_min_length,
    validate_max_length,
    validate_pattern,
    validate_number_range,
    validate_url,
    validate_phone,
    validate_password_strength,
    create_inline_error,
    create_success_indicator,
    create_char_counter,
    create_password_strength_meter,
    get_validation_css,
    create_form_validator,
)


class TestValidationEnums:
    """Tests for validation-related enums."""

    def test_validation_state_values(self):
        """Verify all ValidationState values."""
        assert ValidationState.PRISTINE.value == "pristine"
        assert ValidationState.VALID.value == "valid"
        assert ValidationState.INVALID.value == "invalid"
        assert ValidationState.VALIDATING.value == "validating"

    def test_validation_state_iteration(self):
        """Test ValidationState can be iterated."""
        all_states = list(ValidationState)
        assert len(all_states) == 4


class TestValidationResultDataclass:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid result creation."""
        result = ValidationResult(
            valid=True,
            field_name="email",
            state=ValidationState.VALID
        )
        assert result.valid is True
        assert result.field_name == "email"
        assert result.state == ValidationState.VALID
        assert result.message == ""

    def test_invalid_result(self):
        """Test invalid result creation."""
        result = ValidationResult(
            valid=False,
            message="Email is required",
            field_name="email",
            state=ValidationState.INVALID
        )
        assert result.valid is False
        assert result.message == "Email is required"
        assert result.state == ValidationState.INVALID

    def test_bool_conversion_valid(self):
        """Test ValidationResult truthy when valid."""
        result = ValidationResult(valid=True)
        assert bool(result) is True

    def test_bool_conversion_invalid(self):
        """Test ValidationResult falsy when invalid."""
        result = ValidationResult(valid=False)
        assert bool(result) is False

    def test_result_defaults(self):
        """Test ValidationResult default values."""
        result = ValidationResult(valid=True)
        assert result.message == ""
        assert result.field_name == ""
        assert result.state == ValidationState.PRISTINE


class TestValidationRuleDataclass:
    """Tests for ValidationRule dataclass."""

    def test_rule_creation(self):
        """Test rule creation with all parameters."""
        rule = ValidationRule(
            validator=validate_required,
            message="Field is required",
            skip_empty=False
        )
        assert rule.validator == validate_required
        assert rule.message == "Field is required"
        assert rule.skip_empty is False

    def test_rule_with_skip_empty(self):
        """Test rule with skip_empty enabled."""
        rule = ValidationRule(
            validator=validate_email,
            message="Invalid email",
            skip_empty=True
        )
        assert rule.skip_empty is True


class TestValidateRequired:
    """Tests for validate_required function."""

    def test_required_with_string(self):
        """Test required validation with strings."""
        assert validate_required("hello") is True
        assert validate_required("  ") is False
        assert validate_required("") is False

    def test_required_with_none(self):
        """Test required validation with None."""
        assert validate_required(None) is False

    def test_required_with_list(self):
        """Test required validation with lists."""
        assert validate_required([1, 2, 3]) is True
        assert validate_required([]) is False

    def test_required_with_dict(self):
        """Test required validation with dicts."""
        assert validate_required({"key": "value"}) is True
        assert validate_required({}) is False

    def test_required_with_numbers(self):
        """Test required validation with numbers."""
        assert validate_required(0) is True
        assert validate_required(42) is True
        assert validate_required(0.0) is True

    def test_required_with_bool(self):
        """Test required validation with booleans."""
        assert validate_required(True) is True
        assert validate_required(False) is True  # False is a valid value


class TestValidateEmail:
    """Tests for validate_email function."""

    def test_valid_emails(self):
        """Test valid email addresses."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.org",
            "user+tag@company.co.uk",
            "123@numbers.com",
            "a@b.co",
        ]
        for email in valid_emails:
            assert validate_email(email) is True, f"Should be valid: {email}"

    def test_invalid_emails(self):
        """Test invalid email addresses."""
        invalid_emails = [
            "notanemail",
            "@nodomain.com",
            "noat.com",
            "spaces in@email.com",
            "double@@at.com",
        ]
        for email in invalid_emails:
            assert validate_email(email) is False, f"Should be invalid: {email}"

    def test_empty_email(self):
        """Test empty email passes (handled by required)."""
        assert validate_email("") is True
        assert validate_email(None) is True


class TestValidateMinLength:
    """Tests for validate_min_length function."""

    def test_min_length_pass(self):
        """Test strings meeting minimum length."""
        validator = validate_min_length(5)
        assert validator("hello") is True
        assert validator("hello world") is True

    def test_min_length_fail(self):
        """Test strings below minimum length."""
        validator = validate_min_length(5)
        assert validator("hi") is False
        assert validator("four") is False

    def test_min_length_exact(self):
        """Test string at exact minimum length."""
        validator = validate_min_length(5)
        assert validator("exact") is True  # 5 chars

    def test_min_length_empty(self):
        """Test empty string passes (handled by required)."""
        validator = validate_min_length(5)
        assert validator("") is True
        assert validator(None) is True


class TestValidateMaxLength:
    """Tests for validate_max_length function."""

    def test_max_length_pass(self):
        """Test strings within maximum length."""
        validator = validate_max_length(10)
        assert validator("short") is True
        assert validator("") is True

    def test_max_length_fail(self):
        """Test strings exceeding maximum length."""
        validator = validate_max_length(10)
        assert validator("this is too long") is False

    def test_max_length_exact(self):
        """Test string at exact maximum length."""
        validator = validate_max_length(10)
        assert validator("exactlyten") is True  # 10 chars

    def test_max_length_empty(self):
        """Test empty string passes."""
        validator = validate_max_length(10)
        assert validator("") is True
        assert validator(None) is True


class TestValidatePattern:
    """Tests for validate_pattern function."""

    def test_pattern_match(self):
        """Test strings matching pattern."""
        validator = validate_pattern(r"^\d{5}$")  # 5 digits
        assert validator("12345") is True
        assert validator("00000") is True

    def test_pattern_no_match(self):
        """Test strings not matching pattern."""
        validator = validate_pattern(r"^\d{5}$")
        assert validator("1234") is False
        assert validator("123456") is False
        assert validator("abcde") is False

    def test_pattern_with_flags(self):
        """Test pattern with regex flags."""
        validator = validate_pattern(r"^[a-z]+$", re.IGNORECASE)
        assert validator("ABC") is True
        assert validator("abc") is True

    def test_pattern_empty(self):
        """Test empty string passes."""
        validator = validate_pattern(r"^\d+$")
        assert validator("") is True
        assert validator(None) is True


class TestValidateNumberRange:
    """Tests for validate_number_range function."""

    def test_number_in_range(self):
        """Test numbers within range."""
        validator = validate_number_range(min_val=0, max_val=100)
        assert validator(50) is True
        assert validator(0) is True
        assert validator(100) is True
        assert validator(50.5) is True

    def test_number_below_min(self):
        """Test numbers below minimum."""
        validator = validate_number_range(min_val=0, max_val=100)
        assert validator(-1) is False

    def test_number_above_max(self):
        """Test numbers above maximum."""
        validator = validate_number_range(min_val=0, max_val=100)
        assert validator(101) is False

    def test_number_only_min(self):
        """Test range with only minimum."""
        validator = validate_number_range(min_val=0)
        assert validator(0) is True
        assert validator(1000) is True
        assert validator(-1) is False

    def test_number_only_max(self):
        """Test range with only maximum."""
        validator = validate_number_range(max_val=100)
        assert validator(-1000) is True
        assert validator(100) is True
        assert validator(101) is False

    def test_number_from_string(self):
        """Test number validation from string input."""
        validator = validate_number_range(min_val=0, max_val=100)
        assert validator("50") is True
        assert validator("50.5") is True

    def test_number_invalid_string(self):
        """Test invalid string fails."""
        validator = validate_number_range(min_val=0, max_val=100)
        assert validator("not a number") is False

    def test_number_empty(self):
        """Test empty values pass."""
        validator = validate_number_range(min_val=0, max_val=100)
        assert validator("") is True
        assert validator(None) is True


class TestValidateUrl:
    """Tests for validate_url function."""

    def test_valid_urls(self):
        """Test valid URLs."""
        valid_urls = [
            "http://example.com",
            "https://example.com",
            "https://www.example.com/path",
            "http://sub.domain.example.com",
            "https://example.com/path?query=value",
        ]
        for url in valid_urls:
            assert validate_url(url) is True, f"Should be valid: {url}"

    def test_invalid_urls(self):
        """Test invalid URLs."""
        invalid_urls = [
            "notaurl",
            "ftp://example.com",
            "example.com",
            "//example.com",
        ]
        for url in invalid_urls:
            assert validate_url(url) is False, f"Should be invalid: {url}"

    def test_empty_url(self):
        """Test empty URL passes."""
        assert validate_url("") is True
        assert validate_url(None) is True


class TestValidatePhone:
    """Tests for validate_phone function."""

    def test_valid_phones(self):
        """Test valid phone numbers."""
        valid_phones = [
            "1234567890",
            "+11234567890",
            "123-456-7890",
            "(123) 456-7890",
            "123.456.7890",
            "+1 234 567 8901",
        ]
        for phone in valid_phones:
            assert validate_phone(phone) is True, f"Should be valid: {phone}"

    def test_invalid_phones(self):
        """Test invalid phone numbers."""
        invalid_phones = [
            "123456",  # Too short
            "12345678901234567",  # Too long
            "abcdefghij",
        ]
        for phone in invalid_phones:
            assert validate_phone(phone) is False, f"Should be invalid: {phone}"

    def test_empty_phone(self):
        """Test empty phone passes."""
        assert validate_phone("") is True
        assert validate_phone(None) is True


class TestValidatePasswordStrength:
    """Tests for validate_password_strength function."""

    def test_strong_password(self):
        """Test strong passwords pass."""
        strong_passwords = [
            "Password1",
            "MyP4ssword",
            "Str0ngPass!",
        ]
        for pwd in strong_passwords:
            assert validate_password_strength(pwd) is True, f"Should be strong: {pwd}"

    def test_weak_passwords(self):
        """Test weak passwords fail."""
        weak_passwords = [
            "short",  # Too short
            "alllowercase1",  # No uppercase
            "ALLUPPERCASE1",  # No lowercase
            "NoDigitsHere",  # No digits
            "Pass1",  # Too short
        ]
        for pwd in weak_passwords:
            assert validate_password_strength(pwd) is False, f"Should be weak: {pwd}"

    def test_empty_password(self):
        """Test empty password passes (handled by required)."""
        assert validate_password_strength("") is True
        assert validate_password_strength(None) is True


class TestFieldValidator:
    """Tests for FieldValidator class."""

    def test_simple_validation(self):
        """Test simple field validation."""
        validator = FieldValidator(
            field_name="email",
            rules=[ValidationRule(validate_required, "Email is required")]
        )
        result = validator.validate("test@example.com")
        assert result.valid is True
        assert result.field_name == "email"
        assert result.state == ValidationState.VALID

    def test_validation_fails_first_rule(self):
        """Test validation stops at first failure."""
        validator = FieldValidator(
            field_name="email",
            rules=[
                ValidationRule(validate_required, "Required"),
                ValidationRule(validate_email, "Invalid email"),
            ]
        )
        result = validator.validate("")
        assert result.valid is False
        assert result.message == "Required"

    def test_validation_fails_second_rule(self):
        """Test validation fails on second rule."""
        validator = FieldValidator(
            field_name="email",
            rules=[
                ValidationRule(validate_required, "Required"),
                ValidationRule(validate_email, "Invalid email"),
            ]
        )
        result = validator.validate("notanemail")
        assert result.valid is False
        assert result.message == "Invalid email"

    def test_skip_empty_rule(self):
        """Test skip_empty rule behavior."""
        validator = FieldValidator(
            field_name="website",
            rules=[
                ValidationRule(validate_url, "Invalid URL", skip_empty=True),
            ]
        )
        # Empty should pass
        result = validator.validate("")
        assert result.valid is True

        # Invalid should fail
        result = validator.validate("notaurl")
        assert result.valid is False

    def test_add_rule_chaining(self):
        """Test add_rule method with chaining."""
        validator = FieldValidator(field_name="name")
        result = validator.add_rule(validate_required, "Required")
        assert result is validator  # Returns self for chaining
        assert len(validator.rules) == 1

    def test_multiple_add_rules(self):
        """Test adding multiple rules."""
        validator = FieldValidator(field_name="password")
        validator.add_rule(validate_required, "Required")
        validator.add_rule(validate_min_length(8), "Too short")
        validator.add_rule(validate_password_strength, "Too weak")

        assert len(validator.rules) == 3

    def test_check_required_only(self):
        """Test check_required_only mode."""
        validator = FieldValidator(
            field_name="email",
            rules=[
                ValidationRule(validate_required, "Required"),
                ValidationRule(validate_email, "Invalid email"),
            ]
        )
        # With check_required_only, only required check runs
        result = validator.validate("notanemail", check_required_only=True)
        # Should pass because required is satisfied
        assert result.valid is True


class TestCreateInlineError:
    """Tests for create_inline_error function."""

    def test_error_visible(self):
        """Test visible error message."""
        html = create_inline_error("Email is required", "email", True)
        assert 'class="field-error"' in html
        assert 'role="alert"' in html
        assert "Email is required" in html
        assert 'id="email-error"' in html

    def test_error_hidden(self):
        """Test hidden error message."""
        html = create_inline_error("Error", "field", False)
        assert html == ""

    def test_error_empty_message(self):
        """Test empty error message."""
        html = create_inline_error("", "field", True)
        assert html == ""

    def test_error_no_field_id(self):
        """Test error without field ID."""
        html = create_inline_error("Error message", "", True)
        assert 'id="' not in html or 'id=""' not in html
        assert "Error message" in html


class TestCreateSuccessIndicator:
    """Tests for create_success_indicator function."""

    def test_success_visible(self):
        """Test visible success indicator."""
        html = create_success_indicator("Valid", True)
        assert 'class="field-success"' in html
        assert "Valid" in html

    def test_success_hidden(self):
        """Test hidden success indicator."""
        html = create_success_indicator("Valid", False)
        assert html == ""

    def test_success_custom_message(self):
        """Test custom success message."""
        html = create_success_indicator("Email is valid!", True)
        assert "Email is valid!" in html


class TestCreateCharCounter:
    """Tests for create_char_counter function."""

    def test_char_counter_normal(self):
        """Test character counter in normal state."""
        html = create_char_counter(50, 200)
        assert "50/200" in html
        assert 'class="char-counter"' in html
        assert "warning" not in html
        assert "error" not in html

    def test_char_counter_warning(self):
        """Test character counter in warning state."""
        html = create_char_counter(170, 200)  # 85% - above 80% threshold
        assert "170/200" in html
        assert "warning" in html

    def test_char_counter_error(self):
        """Test character counter in error state."""
        html = create_char_counter(210, 200)  # Over limit
        assert "210/200" in html
        assert "error" in html

    def test_char_counter_at_threshold(self):
        """Test character counter at exact threshold."""
        html = create_char_counter(160, 200)  # Exactly 80%
        assert "160/200" in html
        # At exactly 80%, should show warning
        assert "warning" in html

    def test_char_counter_zero_max(self):
        """Test character counter with zero max."""
        html = create_char_counter(0, 0)
        assert "0/0" in html


class TestCreatePasswordStrengthMeter:
    """Tests for create_password_strength_meter function."""

    def test_empty_password(self):
        """Test meter with empty password."""
        html = create_password_strength_meter("")
        assert 'class="password-strength"' in html
        assert 'class="password-strength-fill"' in html

    def test_weak_password(self):
        """Test meter with weak password."""
        html = create_password_strength_meter("short")
        assert "weak" in html.lower()

    def test_fair_password(self):
        """Test meter with fair password."""
        # "pass1" = 2 criteria: lowercase + digit (not 8+ chars)
        html = create_password_strength_meter("pass1")
        # fair = 2 criteria met
        assert "fair" in html.lower()

    def test_good_password(self):
        """Test meter with good password."""
        html = create_password_strength_meter("Password1")  # 8+, upper, lower, digit
        assert "good" in html.lower()

    def test_strong_password(self):
        """Test meter with strong password."""
        html = create_password_strength_meter("Password1!")  # All criteria
        assert "strong" in html.lower()

    def test_meter_accessibility(self):
        """Test meter accessibility attributes."""
        html = create_password_strength_meter("test")
        assert 'role="meter"' in html
        assert 'aria-label=' in html
        assert 'aria-valuenow=' in html
        assert 'aria-valuemin="0"' in html
        assert 'aria-valuemax="5"' in html


class TestGetValidationCSS:
    """Tests for get_validation_css function."""

    def test_css_returns_string(self):
        """Test CSS function returns string."""
        css = get_validation_css()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_css_matches_constant(self):
        """Test CSS function returns the constant."""
        assert get_validation_css() == VALIDATION_CSS

    def test_css_contains_field_classes(self):
        """Test CSS contains field validation classes."""
        css = get_validation_css()
        assert ".field-valid" in css
        assert ".field-invalid" in css
        assert ".field-error" in css
        assert ".field-success" in css

    def test_css_contains_char_counter(self):
        """Test CSS contains character counter classes."""
        css = get_validation_css()
        assert ".char-counter" in css
        assert ".char-counter.warning" in css
        assert ".char-counter.error" in css

    def test_css_contains_password_meter(self):
        """Test CSS contains password strength meter classes."""
        css = get_validation_css()
        assert ".password-strength" in css
        assert ".password-strength-fill" in css
        assert ".weak" in css
        assert ".fair" in css
        assert ".good" in css
        assert ".strong" in css


class TestCreateFormValidator:
    """Tests for create_form_validator function."""

    def test_form_validator_all_valid(self):
        """Test form validator with all valid fields."""
        validator = create_form_validator({
            "email": FieldValidator("email", [
                ValidationRule(validate_required, "Required"),
                ValidationRule(validate_email, "Invalid email"),
            ]),
            "name": FieldValidator("name", [
                ValidationRule(validate_required, "Required"),
            ]),
        })

        results = validator({
            "email": "test@example.com",
            "name": "John Doe",
        })

        assert results["email"].valid is True
        assert results["name"].valid is True

    def test_form_validator_some_invalid(self):
        """Test form validator with some invalid fields."""
        validator = create_form_validator({
            "email": FieldValidator("email", [
                ValidationRule(validate_required, "Required"),
                ValidationRule(validate_email, "Invalid email"),
            ]),
            "name": FieldValidator("name", [
                ValidationRule(validate_required, "Required"),
            ]),
        })

        results = validator({
            "email": "notvalid",
            "name": "",
        })

        assert results["email"].valid is False
        assert results["email"].message == "Invalid email"
        assert results["name"].valid is False
        assert results["name"].message == "Required"

    def test_form_validator_missing_field(self):
        """Test form validator with missing field."""
        validator = create_form_validator({
            "email": FieldValidator("email", [
                ValidationRule(validate_required, "Required"),
            ]),
        })

        results = validator({})  # No email field

        assert results["email"].valid is False
        assert results["email"].message == "Required"

    def test_form_validator_returns_dict(self):
        """Test form validator returns dictionary of results."""
        validator = create_form_validator({
            "field1": FieldValidator("field1", []),
            "field2": FieldValidator("field2", []),
        })

        results = validator({"field1": "a", "field2": "b"})

        assert isinstance(results, dict)
        assert "field1" in results
        assert "field2" in results
        assert all(isinstance(r, ValidationResult) for r in results.values())


class TestValidationIntegration:
    """Integration tests for validation workflow."""

    def test_complete_form_workflow(self):
        """Test complete form validation workflow."""
        # Create validators for a registration form
        email_validator = FieldValidator("email")
        email_validator.add_rule(validate_required, "Email is required")
        email_validator.add_rule(validate_email, "Please enter a valid email")

        password_validator = FieldValidator("password")
        password_validator.add_rule(validate_required, "Password is required")
        password_validator.add_rule(validate_min_length(8), "Password must be at least 8 characters")
        password_validator.add_rule(validate_password_strength, "Password is too weak")

        form_validator = create_form_validator({
            "email": email_validator,
            "password": password_validator,
        })

        # Test invalid submission
        results = form_validator({"email": "", "password": ""})
        assert not results["email"]
        assert not results["password"]

        # Test valid submission
        results = form_validator({
            "email": "user@example.com",
            "password": "SecurePass1!",
        })
        assert results["email"]
        assert results["password"]

    def test_validation_with_ui_feedback(self):
        """Test validation integrated with UI feedback."""
        validator = FieldValidator("email")
        validator.add_rule(validate_required, "Email is required")
        validator.add_rule(validate_email, "Invalid email format")

        # Validate empty field
        result = validator.validate("")
        if not result:
            error_html = create_inline_error(result.message, result.field_name, True)
            assert "Email is required" in error_html

        # Validate invalid email
        result = validator.validate("invalid")
        if not result:
            error_html = create_inline_error(result.message, result.field_name, True)
            assert "Invalid email format" in error_html

        # Validate valid email
        result = validator.validate("user@example.com")
        if result:
            success_html = create_success_indicator("Valid email", True)
            assert "Valid email" in success_html

    def test_optional_field_validation(self):
        """Test validation of optional fields."""
        # Website is optional but must be valid if provided
        validator = FieldValidator("website")
        validator.add_rule(validate_url, "Invalid URL format", skip_empty=True)

        # Empty is valid
        assert validator.validate("").valid is True

        # Invalid URL fails
        assert validator.validate("notaurl").valid is False

        # Valid URL passes
        assert validator.validate("https://example.com").valid is True
