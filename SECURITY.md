# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in the photonic neuromorphics simulation framework, please report it responsibly.

### How to Report

**Please do NOT create a public GitHub issue for security vulnerabilities.**

Instead, please:

1. **Email us directly**: Send details to daniel@terragon.ai
2. **Include in your report**:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Suggested mitigation (if any)

### What to Expect

- **Acknowledgment**: We'll acknowledge receipt within 48 hours
- **Assessment**: Initial assessment within 5 business days
- **Updates**: Regular updates on our progress
- **Resolution**: We aim to resolve critical issues within 30 days

### Response Process

1. **Triage**: We'll assess the severity and impact
2. **Fix Development**: Create and test security patches
3. **Coordinated Disclosure**: We'll work with you on disclosure timing
4. **Release**: Push security updates to supported versions
5. **Advisory**: Publish security advisory after fixes are available

## Security Considerations

### Simulation Security
- **Input Validation**: All optical parameters are validated for physical bounds
- **File Handling**: SPICE netlists and GDS files are parsed safely
- **Code Generation**: RTL generation includes sanitization of user inputs

### Development Security
- **Dependencies**: We monitor dependencies for known vulnerabilities
- **Code Analysis**: Static analysis tools check for security issues
- **Access Control**: Repository access follows least-privilege principles

### Deployment Security
- **Environment Isolation**: Recommend containerized deployment
- **Secrets Management**: No hardcoded credentials or API keys
- **Network Security**: Document secure communication practices

## Security Best Practices

### For Users
1. **Keep Updated**: Use the latest version with security patches
2. **Validate Inputs**: Sanitize external data before simulation
3. **Secure Environment**: Run simulations in isolated environments
4. **Monitor Dependencies**: Keep Python packages updated

### For Contributors
1. **Code Review**: All changes require security-aware review
2. **Static Analysis**: Use provided security linters
3. **Dependency Updates**: Promptly update vulnerable dependencies
4. **Secret Scanning**: Avoid committing sensitive information

## Known Security Considerations

### SPICE Simulation
- **File Injection**: SPICE netlist parsing could be vulnerable to malicious files
- **Resource Exhaustion**: Large simulations might consume excessive resources
- **Privilege Escalation**: External simulators may require elevated privileges

### RTL Generation
- **Code Injection**: Generated Verilog should be validated before synthesis
- **Tool Chain Security**: Synthesis tools may have their own vulnerabilities
- **Supply Chain**: PDK files and libraries should be from trusted sources

### Mitigation Strategies
- Input validation and sanitization
- Resource limits and timeouts
- Sandboxed execution environments
- Verification of generated outputs

## Vulnerability Disclosure Timeline

- **Day 0**: Vulnerability reported privately
- **Day 1-2**: Acknowledgment sent to reporter
- **Day 3-7**: Initial assessment and triage
- **Day 8-30**: Development and testing of fixes
- **Day 31**: Coordinated public disclosure (if needed)

## Contact Information

- **Security Email**: daniel@terragon.ai
- **PGP Key**: Available upon request
- **Response Time**: Within 48 hours during business days

## Attribution

We believe in responsible disclosure and will acknowledge security researchers who help improve our security posture. With your permission, we'll:

- Credit you in our security advisory
- Mention your contribution in release notes
- Add you to our security hall of fame

Thank you for helping keep the photonic neuromorphics community secure! 🔒